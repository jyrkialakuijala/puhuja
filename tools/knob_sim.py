"""Similarity of puhuja's Kennedy reading to 11labs (gold) and Kokoro, word by word, for the knob searches.

  tools/venv/bin/python -I tools/knob_sim.py eval   --bin BIN [--params FILE] [--seeds 0]       -> one JSON line
  tools/venv/bin/python -I tools/knob_sim.py screen --bin BIN --screen run6.screen.tsv --out ref/opt/ksearch/run7
                                                    [--procs 8] [--shard k/n]

Jyrki 2026-11-18: "run search on parameters to improve whisper scores and general behaviour, similarity of kennedy to
11labs and kokoro". Whisper is judged by knob_search.py / knob_line.py (stt/venv); Zimtohrli lives in tools/venv, so
this is a separate process that knob_line.py --sim calls per point.

Per voice, every word of kennedy.txt is cut from puhuja's render (puhuja -words) and from the references (their
.words.tsv, matched by spelling), each cut level-normalized to -20 dB and padded to 200 ms (zimt_words.cut), and
    d_word = 0.6 * mean(distance to the 11labs readings of that sex)          gold: 11labs1 + 11labs3 (m), 11labs2 (f)
           + 0.4 * weighted best-3 of the distances to six Kokoro voices      (zimt_words.weighted)
The score is 1000 * mean d_word (lower = closer). 11labs gets the larger weight ([[elevenlabs-gold]]). Zimtohrli
aligns in time, so stretching can buy similarity: the searches keep their 1 % length check.

screen: for every candidate of a knob_search screen whose renders were not byte-identical to its base (the dead ones
cannot change the similarity), render both voices with seed 0 and score; rows to <out>.sim.tsv (resumable).
"""
import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knob_search as ks  # noqa: E402
import zimt_words as zw  # noqa: E402

GOLD = {'m': ['11labs1', '11labs3'], 'f': ['11labs2']}
KOKO = {'m': ['am_adam', 'am_onyx', 'am_michael', 'am_eric', 'am_liam', 'bm_george'],
        'f': ['af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_aoede', 'bf_emma']}
W_GOLD = 0.6
import re  # noqa: E402
TIMING = re.compile(r'LEN|DUR|TEMPO|PAUSE|_MS|RATE|SPEED|HOLD|TIMING|Length')  # as knob_line.py
TRACT = re.compile(r'^LINE_V\d+_LEN$|^LINE_LARYNX_LEN$')
TEXT_WORDS = ks.TEXT.replace('–', ' ').replace('—', ' ').split()

_REF = {}
_Z = None


def _cuts(wav, tsv):
    x = zw.read_wav(wav)
    rows = zw.read_tsv(tsv)
    m = zw.match_words(TEXT_WORDS, rows)
    return [zw.cut(x, rows[m[i]][0], rows[m[i]][1]) if i in m else None for i in range(len(TEXT_WORDS))]


def refs(sex):
    if sex not in _REF:
        g = [_cuts(os.path.join(ks.ROOT, 'ref/kennedy/%s.wav' % n), os.path.join(ks.ROOT, 'ref/kennedy/%s.words.tsv' % n)) for n in GOLD[sex]]
        k = [_cuts(os.path.join(ks.ROOT, 'ref/kokoro/kennedy-%s.wav' % n), os.path.join(ks.ROOT, 'ref/kokoro/kennedy-%s.words.tsv' % n)) for n in KOKO[sex]]
        _REF[sex] = (g, k)
    return _REF[sex]


def zim():
    global _Z
    if _Z is None:
        import zimtohrli
        _Z = zimtohrli.Pyohrli()
    return _Z


def score(x, W, sex):
    """-> (score, gold part, kokoro part), each 1000 * mean over the words both sides have"""
    g, k = refs(sex)
    z = zim()
    dg, dk, dw = [], [], []
    for i in range(len(TEXT_WORDS)):
        if i not in W:
            continue
        p = zw.cut(x, W[i][0], W[i][1])
        if p is None:
            continue
        a = [z.distance(p, r[i]) for r in g if r[i] is not None]
        b = [z.distance(p, r[i]) for r in k if r[i] is not None]
        if not a or not b:
            continue
        ga = float(np.mean(a)); kb = zw.weighted(b, 3)
        dg.append(ga); dk.append(kb); dw.append(W_GOLD * ga + (1 - W_GOLD) * kb)
    return 1000 * float(np.mean(dw)), 1000 * float(np.mean(dg)), 1000 * float(np.mean(dk))


def job(binp, params, voice, seed, tmp):
    x, W, md5 = ks.render(binp, params, voice, seed, tmp)
    s, sg, sk = score(x, W, voice)
    return dict(sim=s, gold=sg, koko=sk, n=len(x), md5=md5)


def read_params(path):
    p = {}
    if path:
        for l in open(path):
            if l.strip() and not l.startswith('#'):
                n, v = l.split()[:2]; p[n] = float(v)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['eval', 'screen'])
    ap.add_argument('--bin', required=True)
    ap.add_argument('--params', default=None)
    ap.add_argument('--base-params', default=None)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--screen', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    bp = read_params(a.base_params)
    tmp = tempfile.mkdtemp(prefix='ksim-', dir=os.path.join(ks.ROOT, 'ref/opt/ksearch'))
    try:
        with ProcessPoolExecutor(a.procs) as ex:
            if a.mode == 'eval':
                p = dict(bp, **read_params(a.params))
                fu = {(v, s): ex.submit(job, a.bin, p, v, s, tmp) for v in 'mf' for s in seeds}
                r = {'%s%d' % k: f.result() for k, f in fu.items()}
                out = {v: float(np.mean([r['%s%d' % (v, s)]['sim'] for s in seeds])) for v in 'mf'}
                out.update({v + '_' + q: float(np.mean([r['%s%d' % (v, s)][q] for s in seeds])) for v in 'mf' for q in ('gold', 'koko', 'n')})
                print(json.dumps(out), flush=True)
                return
            # screen
            k, n = map(int, a.shard.split('/'))
            cands = []
            for l in open(a.screen):
                c = l.rstrip('\n').split('\t')
                if c[0] in ('name', 'base') or (c[7] == 'same' and c[14] == 'same'):
                    continue
                if TIMING.search(c[1]) and not TRACT.search(c[1]):
                    continue  # knob_line.py leaves timing knobs out of the gradient
                cands.append((c[0], c[1], float(c[2]), float(c[3])))  # the screen's own candidate value
            cands = sorted(set(cands))[k::n]
            tsv = a.out + '.sim.tsv'
            done = set()
            if os.path.exists(tsv):
                done = {l.split('\t')[0] for l in open(tsv)}
            else:
                with open(tsv, 'w') as f:
                    f.write('name\tknob\tmult\tm_sim\tf_sim\tm_dsim\tf_dsim\tm_gold\tf_gold\tm_koko\tf_koko\tm_len\tf_len\n')
            base = {v: ex.submit(job, a.bin, bp, v, 0, tmp) for v in 'mf'}
            base = {v: f.result() for v, f in base.items()}
            print('base m %.3f f %.3f' % (base['m']['sim'], base['f']['sim']), flush=True)
            todo = [c for c in cands if c[0] not in done]
            futs = []
            for name, knob, mult, val in todo:
                p = dict(bp, **{knob: val})
                futs.append((name, knob, mult, {v: ex.submit(job, a.bin, p, v, 0, tmp) for v in 'mf'}))
            for name, knob, mult, fu in futs:
                r = {v: f.result() for v, f in fu.items()}
                with open(tsv, 'a') as f:
                    f.write('%s\t%s\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                        name, knob, mult, r['m']['sim'], r['f']['sim'], r['m']['sim'] - base['m']['sim'], r['f']['sim'] - base['f']['sim'],
                        r['m']['gold'], r['f']['gold'], r['m']['koko'], r['f']['koko'],
                        r['m']['n'] / base['m']['n'], r['f']['n'] / base['f']['n']))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
