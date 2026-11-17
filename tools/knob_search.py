"""One knob at a time, +-10 % / +-5 %, judged by Whisper on the Kennedy reading and by sound checks.

  stt/venv/bin/python -I tools/knob_search.py screen [--out ref/opt/ksearch/run1] [--bin BIN] [--seeds 0,11]
  stt/venv/bin/python -I tools/knob_search.py base   [--out ...] --seeds 21,22,...   (noise of the base)
  stt/venv/bin/python -I tools/knob_search.py confirm [--out ...] --top 40 --seeds 31,...,38
  stt/venv/bin/python -I tools/knob_search.py eval   [--out ...] --params FILE --seeds ...  (any knob set)

Jyrki 2026-11-16: "let's make a search on trying to find better parameters by +- 10 % or +- 5 % search on values
and looking at whisper and other scores".

The knob pool is every knob the Kennedy text reads (PUHUJA_PARAM_LOG, params.h) whose default is not 0, -1 or a
sentinel (|v| >= 900). A candidate is one knob times 0.9 / 1.1 (screen) or 0.9 / 0.95 / 1.05 / 1.1 (confirm). Each
candidate renders kennedy.txt in both voices with PUHUJA_SEED per seed, cuts it into its sentences (puhuja -words),
and Whisper (medium.en, one model kept loaded) transcribes each sentence; the score is word errors per voice, the
same as the round agents' clip test (kwt/clips.py). A render that is byte-identical to the base's for that voice
and seed gets the base's errors without transcribing (a dead knob costs only its renders).

Sound checks on the whole render, against the base with the same seed (the 2026-11-15 regression, VQCS: Whisper
got better while the sound got worse - 'strange bass sounds', 'an AM radio that is not tuned right'):
level (active-frame RMS, dB), peak, bangs (5 ms frames 15 dB over their 105 ms median and over -30 dB), bass
(20-110 Hz energy relative to all, dB). Rows go to <out>.tsv; restarts skip finished candidates.
"""
import argparse
import hashlib
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import stt_device  # noqa: E402

TEXT = open(os.path.join(ROOT, 'kennedy.txt'), encoding='utf-8').read()


# --- the Kennedy text's sentences (as kwt/clips.py: tokens split at whitespace / dashes, a sentence ends in . ! ?)
def canon_tokens(t):
    toks = []
    for m in re.finditer(r'[^\s–—]+([\s–—]*)', t.strip()):
        core = m.group(0).rstrip().strip()
        toks.append((core, bool(core) and core[-1] in '.!?'))
    return toks


TOKS = canon_tokens(TEXT)
SENTS = []
_cur = []
for _k, (_w, _end) in enumerate(TOKS):
    _cur.append(_k)
    if _end:
        SENTS.append(_cur); _cur = []
if _cur:
    SENTS.append(_cur)


def words(t):
    t = t.lower().replace('’', "'")
    return [w for w in re.split(r"[^a-z0-9']+", t) if w]


REFW = [[w for k in ks for w in words(TOKS[k][0])] for ks in SENTS]


def ed(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


# --- rendering
def render(binp, params, voice, seed, tmp):
    """-> (samples float32 48 kHz, word timings {token index: (a, b)}, md5)"""
    tag = hashlib.md5(('%s|%s|%d' % (sorted(params.items()), voice, seed)).encode()).hexdigest()[:12]
    wav = os.path.join(tmp, tag + '.wav'); wt = os.path.join(tmp, tag + '.tsv'); pf = os.path.join(tmp, tag + '.params')
    with open(pf, 'w') as f:
        for k, v in params.items():
            f.write('%s %.9g\n' % (k, v))
    env = dict(os.environ, PUHUJA_PARAMS=pf)
    env.pop('PUHUJA_PARAM_LOG', None)
    if seed:
        env['PUHUJA_SEED'] = str(seed)
    cmd = ['nice', '-n', '10', binp, '-en'] + (['-female'] if voice == 'f' else []) + ['-o', wav, '-words', wt, TEXT]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = open(wav, 'rb').read()
    x = read_wav(raw)
    W = {}
    for l in open(wt):
        a, b, n = l.split()
        k = int(n) - 1
        W[k] = (float(a), float(b)) if k not in W else (W[k][0], float(b))
    for p in (wav, wt, pf):
        os.remove(p)
    return x, W, hashlib.md5(raw).hexdigest()


def read_wav(data):
    pos, fmt = 12, 1
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]; n = int.from_bytes(data[pos + 4:pos + 8], 'little')
        if cid == b'fmt ':
            fmt = int.from_bytes(data[pos + 8:pos + 10], 'little')
        if cid == b'data':
            d = data[pos + 8:pos + 8 + n]
            return np.frombuffer(d, np.float32).copy() if fmt == 3 else np.frombuffer(d, np.int16) / 32768.0
        pos += 8 + n + (n & 1)
    raise ValueError('no data chunk')


def clips(x, W, fs=48000):
    out = []
    for si, ks in enumerate(SENTS):
        ks = [k for k in ks if k in W]
        if not ks:
            out.append(None); continue
        a = max(W[ks[0]][0] - 0.1, 0); b = W[ks[-1]][1] + 0.25
        prv = [W[k][1] for k in range(ks[0]) if k in W]; nxt = [W[k][0] for k in range(ks[-1] + 1, ks[-1] + 4) if k in W]
        if prv: a = max(a, prv[-1] + 0.03)
        if nxt: b = min(b, nxt[0] - 0.03)
        out.append(x[int(a * fs):int(b * fs)])
    return out


def median_filter(r, n):
    p = n // 2
    return np.median(sliding_window_view(np.pad(r, p, mode='edge'), n), axis=1)


def band_db(x, lo=20.0, hi=110.0, fs=48000):
    """energy in lo..hi Hz relative to all, dB (Parseval on one FFT)"""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return 10 * np.log10(X[(f >= lo) & (f <= hi)].sum() / (X.sum() + 1e-30) + 1e-30)


_N = np.arange(-96, 97)
_FIR = (np.sinc(_N / 3 * 0.9) * 0.9 / 3 * np.blackman(len(_N))).astype(np.float32)


def to16k(c):
    return np.convolve(c.astype(np.float32), _FIR, mode='same')[::3].astype(np.float32)


def sound(x):
    x = x.astype(np.float64)
    fr = x[:len(x) // 240 * 240].reshape(-1, 240)
    r = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-7)
    act = r > -40
    e = lambda z: 10 * np.log10((z ** 2).mean() + 1e-20)
    return dict(level=float(10 * np.log10((fr[act] ** 2).mean() + 1e-20)), peak=float(abs(x).max()),
                bangs=int(((r - median_filter(r, 21) > 15) & (r > -30)).sum()), bass=float(band_db(x)))


# --- Whisper
class Judge:
    def __init__(self, model='medium.en'):
        from faster_whisper import WhisperModel
        device, ctype = stt_device.pick('auto')
        self.m = WhisperModel(model, device=device, compute_type=ctype, download_root=os.path.join(ROOT, 'stt', 'models'))

    def errors(self, cl):
        tot = 0
        for si, c in enumerate(cl):
            if c is None or len(c) < 4800:
                tot += len(REFW[si]); continue
            y = to16k(c)
            segs, _ = self.m.transcribe(y, beam_size=5, initial_prompt="Hello, everyone. Welcome back! Today, we're talking about demos.",
                                        condition_on_previous_text=False)
            tot += ed(REFW[si], words(' '.join(s.text for s in segs)))
        return tot


# --- guard words (Jyrki 2026-11-17: the dictionary words that got worse in a Kennedy-only search are read in every
# later one, so they cannot get worse again): rendered one per line as the gate does (whisper_batch.render), heard
# in one batched pass; their wrong words add to the voice's errors (and are kept apart in the _werr columns).
def render_words(binp, params, voice, seed, lines, tmp):
    import whisper_batch as WB
    d = tempfile.mkdtemp(prefix='w', dir=tmp)
    pf = os.path.join(d, 'p.params')
    with open(pf, 'w') as f:
        for k, v in params.items():
            f.write('%s %.9g\n' % (k, v))
    env = dict(os.environ, PUHUJA_PARAMS=pf); env.pop('PUHUJA_PARAM_LOG', None)
    if seed:
        env['PUHUJA_SEED'] = str(seed)
    clips = WB.render(binp, lines, voice, 2, d, env)
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    h = hashlib.md5()
    for c in clips:
        h.update(np.asarray(c, np.float32).tobytes())
    return clips, h.hexdigest()


# --- the pool
def pool(binp, out):
    lp = out + '.knobs'
    if not os.path.exists(lp):
        for v in 'mf':
            env = dict(os.environ, PUHUJA_PARAM_LOG=lp + '.' + v); env.pop('PUHUJA_PARAMS', None)
            subprocess.run([binp, '-en'] + (['-female'] if v == 'f' else []) + ['-r', TEXT], env=env, check=True,
                           stdout=subprocess.DEVNULL)
        d = {}
        for v in 'mf':
            for l in open(lp + '.' + v):
                n, x = l.split(); d.setdefault(n, float(x))
            os.remove(lp + '.' + v)
        with open(lp, 'w') as f:
            for n in sorted(d):
                f.write('%s %.9g\n' % (n, d[n]))
    d = {}
    for l in open(lp):
        n, x = l.split(); d[n] = float(x)
    return {n: x for n, x in d.items() if x != 0 and x != -1 and abs(x) < 900}


class Runner:
    def __init__(self, a):
        self.a = a
        self.tmp = tempfile.mkdtemp(prefix='ks-', dir=os.path.dirname(a.out) or '.')
        self.judge = Judge()
        self.wlines = None
        if getattr(a, 'words', None):
            import whisper_batch as WB
            import whisper_phonemes as WP
            from faster_whisper import BatchedInferencePipeline
            self.WB = WB
            self.wlines = [l.strip() for l in open(a.words, encoding='utf-8') if l.strip()]
            self.wref = [' '.join(WP.norm_word(l)) for l in self.wlines]
            self.pipe = BatchedInferencePipeline(model=self.judge.m)
        self.base = {}  # (voice, seed) -> (md5, errors, sound)
        self.ex = ThreadPoolExecutor(a.procs)
        self.cur = 'base'
        self.bp = {}
        if getattr(a, 'base_params', None):
            for l in open(a.base_params):
                if l.strip() and not l.startswith('#'):
                    n, x = l.split()[:2]; self.bp[n] = float(x)

    def evaluate(self, params, seeds, name):
        """-> {voice: (errors, sound deltas summed over seeds)} for one candidate, pipelined renders"""
        return self.collect(self.submit(params, seeds))

    def submit(self, params, seeds):
        p = dict(self.bp, **params)
        jobs = []
        for v in 'mf':
            for s in seeds:
                fu = self.ex.submit(render, self.a.bin, p, v, s, self.tmp)
                fw = self.ex.submit(render_words, self.a.bin, p, v, s, self.wlines, self.tmp) if self.wlines else None
                jobs.append((v, s, (fu, fw)))
        return jobs

    def word_errors(self, clips):
        audio, starts, chunks = self.WB.concatenate(clips)
        heard = self.WB.transcribe(self.pipe, audio, starts, chunks, 16)
        return sum(h != w for h, w in zip(heard, self.wref))

    def collect(self, jobs):
        res = {}
        for v, s, (fu, fw) in jobs:
            x, W, md5 = fu.result()
            snd = sound(x)
            b = self.base.get((v, s))
            if b and b[0].split(':')[0] == md5:
                ek = b[1] - b[3]
            else:
                ek = self.judge.errors(clips(x, W))
            ew, wmd5 = 0, ''
            if fw is not None:
                wc, wmd5 = fw.result()
                ew = b[3] if b and b[4] == wmd5 else self.word_errors(wc)
                md5 = md5 + ':' + wmd5[:12]
            e = ek + ew
            res[(v, s)] = (md5, e, snd, ew, wmd5)
            with open(self.a.out + '.detail.tsv', 'a') as f:
                f.write('%s\t%s\t%d\t%d\t%s\t%.2f\t%.3f\t%d\t%.2f\t%d\n' % (self.cur, v, s, e, md5[:10], snd['level'], snd['peak'], snd['bangs'], snd['bass'], ew))
        return res

    def ensure_base(self, seeds):
        need = [s for s in seeds if ('m', s) not in self.base]
        if need:
            r = self.evaluate({}, need, 'base')
            self.base.update(r)


def row(name, knob, mult, val, seeds, res, base):
    out = [name, knob, '%.3f' % mult, '%.9g' % val, ','.join(map(str, seeds))]
    for v in 'mf':
        e = sum(res[(v, s)][1] for s in seeds); eb = sum(base[(v, s)][1] for s in seeds)
        same = all(res[(v, s)][0] == base[(v, s)][0] for s in seeds)
        d = {k: np.mean([res[(v, s)][2][k] - base[(v, s)][2][k] for s in seeds]) for k in ('level', 'peak', 'bangs', 'bass')}
        out += [str(e), str(e - eb), 'same' if same else 'diff', '%.2f' % d['level'], '%.3f' % d['peak'], '%.1f' % d['bangs'], '%.2f' % d['bass']]
    for v in 'mf':  # guard words, already inside err / derr above
        w = sum(res[(v, s)][3] for s in seeds); wb = sum(base[(v, s)][3] for s in seeds)
        out += [str(w), str(w - wb)]
    return '\t'.join(out)


HEAD = 'name\tknob\tmult\tvalue\tseeds\t' + '\t'.join('%s_%s' % (v, k) for v in 'mf' for k in ('err', 'derr', 'same', 'dlevel', 'dpeak', 'dbangs', 'dbass')) + '\tm_werr\tm_dwerr\tf_werr\tf_dwerr'


def done_names(path):
    if not os.path.exists(path):
        return set()
    return {l.split('\t')[0] for l in open(path) if l.strip() and not l.startswith('name\t')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['screen', 'base', 'confirm', 'eval'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'ref/opt/ksearch/run1'))
    ap.add_argument('--bin', default=None)
    ap.add_argument('--seeds', default='0,11')
    ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--steps', default=None, help='multipliers, e.g. 0.9,1.1')
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--params', default=None)
    ap.add_argument('--shuffle', type=int, default=1)
    ap.add_argument('--base-params', default=None, help='start point: every candidate (and the base) renders with these values, a candidate multiplies its knob\'s value here')
    ap.add_argument('--skip-dead', default=None, help='a screen tsv: leave out knobs that were byte-identical to its base both ways')
    ap.add_argument('--words', default=None, help='guard word list (one per line), read in both voices and every seed; its wrong words add to the errors')
    ap.add_argument('--shard', default='0/1', help='k/n: this process takes every n-th candidate (two share the GPU)')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.bin is None:
        a.bin = a.out + '.bin'
        if not os.path.exists(a.bin):
            import shutil
            shutil.copy2(os.path.join(ROOT, 'puhuja'), a.bin)
    seeds = [int(s) for s in a.seeds.split(',')]
    R = Runner(a)
    tsv = a.out + '.' + a.mode + '.tsv'
    new = not os.path.exists(tsv)
    f = open(tsv, 'a')
    if new:
        f.write(HEAD + '\n'); f.flush()
    done = done_names(tsv)
    t0 = time.time()
    R.ensure_base(seeds)
    f.write(row('base', '-', 1, 0, seeds, R.base, R.base) + '\n'); f.flush()
    if a.mode == 'base':
        return
    if a.mode == 'eval':
        p = {}
        for l in open(a.params):
            if l.strip() and not l.startswith('#'):
                n, x = l.split()[:2]; p[n] = float(x)
        R.cur = os.path.basename(a.params)
        res = R.evaluate(p, seeds, 'eval')
        f.write(row(os.path.basename(a.params), '*', 1, 0, seeds, res, R.base) + '\n'); f.flush()
        return
    P = pool(a.bin, a.out)
    P = {n: R.bp.get(n, x) for n, x in P.items()}
    if a.skip_dead:
        st = {}
        for l in open(a.skip_dead):
            c = l.rstrip('\n').split('\t')
            if c[0] not in ('name', 'base'):
                st.setdefault(c[1], []).append(c[7] == 'same' and c[14] == 'same')
        P = {n: x for n, x in P.items() if not (n in st and all(st[n]))}
    if a.mode == 'screen':
        steps = [float(s) for s in (a.steps or '0.9,1.1').split(',')]
        cands = [(n, m) for n in sorted(P) for m in steps]
        random.Random(a.shuffle).shuffle(cands)
    else:  # confirm: the best screened knob-directions, every step on both sides of 1 in that direction
        steps = [float(s) for s in (a.steps or '0.9,0.95,1.05,1.1').split(',')]
        sc = {}
        for l in open(a.out + '.screen.tsv'):
            c = l.rstrip('\n').split('\t')
            if c[0] in ('name', 'base'):
                continue
            dm, df = int(c[6]), int(c[13])
            ok = c[7] == 'diff' or c[14] == 'diff'
            if ok:
                sc[(c[1], float(c[2]))] = dm + df
        best = sorted(sc.items(), key=lambda kv: kv[1])[:a.top]
        cands = []
        for (n, m), _ in best:
            for s in steps:
                if (s < 1) == (m < 1) and (n, s) not in cands:
                    cands.append((n, s))
    print('%d candidates, %d seeds, %d done' % (len(cands), len(seeds), len(done)), flush=True)
    # pipeline: keep the render pool full a few candidates ahead of the GPU
    sk, sn = map(int, a.shard.split('/'))
    cands = cands[sk::sn]
    todo = [(n, m) for n, m in cands if '%s*%.3f' % (n, m) not in done]
    ahead = max(2, a.procs // (2 * len(seeds)) + 1)
    q = []
    i = 0
    while i < len(todo) or q:
        while i < len(todo) and len(q) < ahead:
            n, m = todo[i]; i += 1
            p = {n: P[n] * m}
            q.append((n, m, p, R.submit(p, seeds)))
        n, m, p, jobs = q.pop(0)
        R.cur = '%s*%.3f' % (n, m)
        res = R.collect(jobs)
        f.write(row('%s*%.3f' % (n, m), n, m, p[n], seeds, res, R.base) + '\n'); f.flush()
        k = len(done_names(tsv))
        el = time.time() - t0
        print('%5d / %d  %s  %.0f s/cand' % (k, len(cands), '%s*%.3f' % (n, m), el / max(1, i - len(q))), flush=True)


if __name__ == '__main__':
    main()
