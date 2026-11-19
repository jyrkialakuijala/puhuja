"""Line search along the screened gradient, dropping dimensions (Jyrki's plan, 2026-11-16).

  stt/venv/bin/python -I tools/knob_line.py [--out ref/opt/ksearch/run1] [--seeds 41,...,48] [--keep 0.25,0.5]
      [--validate 51,...,66]

Jyrki 2026-11-16: "one strategy could be to go first linearly towards the gradient then dropping 75 % of the
dimensions and going in that direction as long as optimum improves, and again reducing 50 % of the dimensions in
future steps".

Gradient g_k (Whisper word errors per seed, both voices summed, per +10 % of knob k) from <out>.screen.tsv:
the gain on the better side of x0.9 / x1.1 per seed and 10 % step (0 if both
sides are worse: the knob is at its optimum); for knobs in <out>.confirm.tsv the 8-seed least-squares slope over its steps
replaces it. Left out: knobs whose screen renders were byte-identical to the base in both directions (dead), and
knobs whose screen moved level or bass over 0.3 dB in either voice (the VQCS kind of sound change).

Stage 0 uses every remaining knob; each later stage keeps the largest-|g| share of the previous stage's knobs
(--keep: 0.25 first, then 0.5 repeatedly) until <= 2 are left. In a stage the point is
    value_k = v0_k * (1 - alpha * 0.1 * g_k / max|g|)      (v0: the stage's start point; the steepest knob moves
                                                             10 % at alpha 1)
with alpha 0.25, 0.5, 1, 2, 4, ... while the score improves (0.125 is tried if 0.25 fails). Score: m + f word errors
over --seeds (8 per voice). A point is better only if its total is lower than the best so far, neither voice is
worse than the best by over 20, and the sound checks against the base pass (level not lower by > 0.3 dB, bass not
higher by > 0.3 dB, mean peak not > 1.3x, bangs not > 1.5x + 5). The best point carries into the next stage.

Every point goes to <out>.line.tsv; the best params so far to <out>.line.best.params. At the end the best and the
base are re-scored on --validate seeds (fresh: the stage choices were made on --seeds, so the best is flattered),
and two-paragraph listening clips go to <out>.listen/. Resumable: points already in line.tsv are not re-rendered.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knob_search as ks  # noqa: E402


def sim_screen(path):
    """knob_sim.py screen rows -> {knob: {mult: m_dsim + f_dsim}}"""
    S = {}
    for l in open(path):
        c = l.rstrip('\n').split('\t')
        if c[0] == 'name':
            continue
        S.setdefault(c[1], {})[float(c[2])] = float(c[5]) + float(c[6])
    return S


def sim_lambda(out, S):
    """errors per seed per similarity unit: the screens' typical knob effects (median |delta|) made equal"""
    w, s = [], []
    for l in open(out + '.screen.tsv'):
        c = l.rstrip('\n').split('\t')
        if c[0] in ('name', 'base') or c[1] not in S or float(c[2]) not in S[c[1]]:
            continue
        w.append(abs(int(c[6]) + int(c[13])) / len(c[4].split(',')))
        s.append(abs(S[c[1]][float(c[2])]))
    return float(np.median(w) / max(np.median(s), 1e-6))


def gradient(out, S=None, lam=0.0):
    E = {}
    snd = {}
    for l in open(out + '.screen.tsv'):
        c = l.rstrip('\n').split('\t')
        if c[0] in ('name', 'base'):
            continue
        k, m = c[1], float(c[2])
        dead = c[7] == 'same' and c[14] == 'same'
        e = int(c[6]) + int(c[13])
        if S is not None:  # combined: per-seed errors (screen seeds) + lam * similarity, scaled back to the screen's sum
            ns = len(c[4].split(','))
            e = e + lam * ns * S.get(k, {}).get(m, 0.0)
        E.setdefault(k, {})[m] = (e, dead)
        s = max(abs(float(c[8])), abs(float(c[15])), float(c[11]), float(c[18]))
        snd[k] = max(snd.get(k, 0), s)
    g = {}
    for k, d in E.items():
        if 0.9 in d and 1.1 in d and not (d[0.9][1] and d[1.1][1]) and snd[k] <= 0.3:
            # the gain on the better side, not the central difference: a cliff on one side (LINE_Z_X x1.1: +134
            # errors, x0.9: +13) would otherwise read as a steep slope toward a value that is itself worse; a knob
            # worse both ways is at its optimum here (g 0)
            b = min(d[1.1][0], d[0.9][0])
            if b < 0:
                g[k] = (-b / 2 / 2) * (1 if d[0.9][0] < d[1.1][0] else -1)
    if S is None and os.path.exists(out + '.confirm.tsv'):
        pts = {}
        for l in open(out + '.confirm.tsv'):
            c = l.rstrip('\n').split('\t')
            if c[0] in ('name', 'base'):
                continue
            nseed = len(c[4].split(','))
            pts.setdefault(c[1], []).append(((float(c[2]) - 1) / 0.1, (int(c[6]) + int(c[13])) / nseed,
                                             max(abs(float(c[8])), abs(float(c[15])), float(c[11]), float(c[18]))))
        for k, p in pts.items():
            x = np.array([a for a, _, _ in p]); y = np.array([b for _, b, _ in p])
            if max(s for _, _, s in p) > 0.3:
                g.pop(k, None); continue
            g[k] = float((x * y).sum() / (x * x).sum())
    return {k: v for k, v in g.items() if v != 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ks.ROOT, 'ref/opt/ksearch/run1'))
    ap.add_argument('--bin', default=None)
    ap.add_argument('--seeds', default='41,42,43,44,45,46,47,48')
    ap.add_argument('--validate', default='51,52,53,54,55,56,57,58,61,62,63,64,65,66,67,68')
    ap.add_argument('--keep', default='0.25,0.5')
    ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--words', default=None, help='guard word list, as knob_search.py --words (counted in the score)')
    ap.add_argument('--steps-from', default=None, help='screen tsv to take the gradient from (default <out>.screen.tsv)')
    ap.add_argument('--sim', default=None, help='knob_sim.py screen tsv: the objective becomes Whisper errors + lam * nseeds * (m + f '
                    'Zimtohrli similarity to 11labs / Kokoro), each point scored by knob_sim.py eval as well')
    ap.add_argument('--sim-lambda', type=float, default=None, help='errors per seed per similarity unit (default: from the screens)')
    ap.add_argument('--sim-seeds', default='0,11')
    ap.add_argument('--with-timing', action='store_true', help='let timing knobs (durations, tempo, pauses, kVowelLength*) into the gradient')
    a = ap.parse_args()
    a.bin = a.bin or a.out + '.bin'
    seeds = [int(s) for s in a.seeds.split(',')]
    keep = [float(x) for x in a.keep.split(',')]
    P = ks.pool(a.bin, a.out)
    gsrc = a.steps_from[:-len('.screen.tsv')] if a.steps_from else a.out
    S = sim_screen(a.sim) if a.sim else None
    lam = 0.0
    if S is not None:
        lam = a.sim_lambda if a.sim_lambda is not None else sim_lambda(gsrc, S)
        print('similarity weight: %.3f errors per seed per unit' % lam, flush=True)
    g = {k: v for k, v in gradient(gsrc, S, lam).items() if k in P}
    if not a.with_timing:
        # Timing knobs stay out: slower speech pleases Whisper (run1: half the male gain; run5: kVowelLength50 alone
        # +4.7 % length), and one of them in the gradient would trip the 1 % length check at every step. The
        # vocal-tract lengths (LINE_V<n>_LEN, LINE_LARYNX_LEN) are timbre, not timing.
        tre = re.compile(r'LEN|DUR|TEMPO|PAUSE|_MS|RATE|SPEED|HOLD|TIMING|Length')
        tract = re.compile(r'^LINE_V\d+_LEN$|^LINE_LARYNX_LEN$')
        ng = len(g)
        g = {k: v for k, v in g.items() if not tre.search(k) or tract.search(k)}
        print('%d timing knobs left out of the gradient' % (ng - len(g)), flush=True)
    R = ks.Runner(a)
    tsv = a.out + '.line.tsv'
    seen = {}
    if os.path.exists(tsv):
        for l in open(tsv):
            c = l.rstrip('\n').split('\t')
            if c[0] != 'stage':
                seen[(int(c[0]), float(c[2]))] = c
    f = open(tsv, 'a')
    if not seen:
        f.write('stage\tndims\talpha\tm_err\tf_err\ttotal\tdlevel_m\tdlevel_f\tdbass_m\tdbass_f\tpeak_x\tbangs\tbase_bangs\tverdict\tm_sim\tf_sim\tobj\n')
    sim_seeds = a.sim_seeds
    pdir = a.out + '.simp'
    os.makedirs(pdir, exist_ok=True)

    def sim_eval(params, tag):
        import json
        import subprocess
        pf = os.path.join(pdir, tag + '.params')
        with open(pf, 'w') as o:
            for k in sorted(params):
                o.write('%s %.9g\n' % (k, params[k]))
        r = subprocess.run([os.path.join(ks.ROOT, 'tools/venv/bin/python'), '-I', '-W', 'ignore', os.path.join(ks.ROOT, 'tools/knob_sim.py'),
                            'eval', '--bin', a.bin, '--params', pf, '--seeds', sim_seeds, '--procs', '4'],
                           check=True, capture_output=True, text=True)
        return json.loads(r.stdout.strip().split('\n')[-1])
    from concurrent.futures import ThreadPoolExecutor
    SX = ThreadPoolExecutor(2)
    bsim = sim_eval({}, 'base') if S is not None else None
    if bsim:
        print('base similarity m %.3f f %.3f' % (bsim['m'], bsim['f']), flush=True)
    R.ensure_base(seeds)
    bm = sum(R.base[('m', s)][1] for s in seeds); bf = sum(R.base[('f', s)][1] for s in seeds)
    bpk = np.mean([R.base[(v, s)][2]['peak'] for v in 'mf' for s in seeds])
    bbg = sum(R.base[(v, s)][2]['bangs'] for v in 'mf' for s in seeds)
    nsd = len(seeds)
    best = dict(params={}, m=bm, f=bf, sm=bsim['m'] if bsim else 0.0, sf=bsim['f'] if bsim else 0.0)
    obj = lambda m, fe, sm, sf: m + fe + lam * nsd * (sm + sf)
    print('base m %d f %d on %s; %d knobs in the gradient' % (bm, bf, a.seeds, len(g)), flush=True)

    def score(params, stage, alpha, nd):
        R.cur = 'line-s%d-a%g' % (stage, alpha)
        sfu = SX.submit(sim_eval, params, R.cur) if S is not None else None
        res = R.evaluate(params, seeds, R.cur)
        sim = sfu.result() if sfu else {'m': 0.0, 'f': 0.0}
        m = sum(res[('m', s)][1] for s in seeds); fe = sum(res[('f', s)][1] for s in seeds)
        d = lambda v, k: np.mean([res[(v, s)][2][k] - R.base[(v, s)][2][k] for s in seeds])
        pk = np.mean([res[(v, s)][2]['peak'] for v in 'mf' for s in seeds]) / bpk
        bg = sum(res[(v, s)][2]['bangs'] for v in 'mf' for s in seeds)
        sound_ok = d('m', 'level') > -0.3 and d('f', 'level') > -0.3 and d('m', 'bass') < 0.3 and d('f', 'bass') < 0.3 \
            and pk < 1.3 and bg < 1.5 * bbg + 5
        # the duration cheat (run5 2026-11-18: kVowelLength50 read the text 4.7 % slower and won the male's Whisper gain)
        slow = np.mean([res[(v, s)][2]['n'] / R.base[(v, s)][2]['n'] for v in 'mf' for s in seeds])
        if slow > 1.01:
            print('  length x%.3f: rejected' % slow, flush=True)
            sound_ok = False
        better = obj(m, fe, sim['m'], sim['f']) < obj(best['m'], best['f'], best['sm'], best['sf']) and m < best['m'] + 20 \
            and fe < best['f'] + 20 and sound_ok
        if S is not None:
            # both objectives: Whisper not worse than the base by more than its noise (8 seeds x 2 voices, sd ~14),
            # similarity not worse than the base in either voice (seed sd ~0.08)
            better = better and m + fe <= bm + bf + 10 and sim['m'] <= bsim['m'] + 0.1 and sim['f'] <= bsim['f'] + 0.1
        verdict = 'better' if better else ('sound' if not sound_ok else 'worse')
        f.write('%d\t%d\t%g\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%s\t%.3f\t%.3f\t%.1f\n' % (
            stage, nd, alpha, m, fe, m + fe, d('m', 'level'), d('f', 'level'), d('m', 'bass'), d('f', 'bass'), pk, bg, bbg, verdict,
            sim['m'], sim['f'], obj(m, fe, sim['m'], sim['f'])))
        f.flush()
        print('stage %d dims %d alpha %g: m %d f %d total %d sim m %.3f f %.3f obj %.1f (best %.1f) %s' % (
            stage, nd, alpha, m, fe, m + fe, sim['m'], sim['f'], obj(m, fe, sim['m'], sim['f']),
            obj(best['m'], best['f'], best['sm'], best['sf']), verdict), flush=True)
        return better, m, fe, sim

    dims = sorted(g, key=lambda k: -abs(g[k]))
    stage = 0
    while dims:
        gm = max(abs(g[k]) for k in dims)
        start = dict(best['params'])

        def point(alpha):
            p = dict(start)
            for k in dims:
                v0 = start.get(k, P[k])
                p[k] = v0 * (1 - alpha * 0.1 * g[k] / gm)
            return p
        alphas = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
        improved = False
        for i, al in enumerate(alphas):
            p = point(al)
            ok, m, fe, sm = score(p, stage, al, len(dims))
            if ok:
                best.update(params=p, m=m, f=fe, sm=sm['m'], sf=sm['f']); improved = True
                with open(a.out + '.line.best.params', 'w') as bf_:
                    bf_.write('# stage %d alpha %g m %d f %d (base m %d f %d, seeds %s)\n' % (stage, al, m, fe, bm, bf, a.seeds))
                    for k in sorted(p):
                        bf_.write('%s %.9g\n' % (k, p[k]))
            elif improved or i > 0:
                break
            elif al == 0.25:
                ok, m, fe, sm = score(point(0.125), stage, 0.125, len(dims))
                if ok:
                    best.update(params=point(0.125), m=m, f=fe, sm=sm['m'], sf=sm['f']); improved = True
                break
        if len(dims) <= 2:
            break
        frac = keep[0] if stage == 0 else keep[-1]
        dims = dims[:max(2, int(round(len(dims) * frac)))]
        stage += 1

    # validation on fresh seeds
    vs = [int(s) for s in a.validate.split(',')]
    R.ensure_base(vs)
    R.cur = 'line-validate'
    res = R.evaluate(best['params'], vs, R.cur)
    vm = sum(res[('m', s)][1] for s in vs); vf = sum(res[('f', s)][1] for s in vs)
    vbm = sum(R.base[('m', s)][1] for s in vs); vbf = sum(R.base[('f', s)][1] for s in vs)
    msg = 'validation on %d fresh seeds: base m %d f %d -> best m %d f %d (%+d / %+d); best changes %d knobs' % (
        len(vs), vbm, vbf, vm, vf, vm - vbm, vf - vbf, len(best['params']))
    if S is not None:
        sb = sim_eval(best['params'], 'validate')
        msg += '; similarity (seeds %s) base m %.3f f %.3f -> best m %.3f f %.3f' % (sim_seeds, bsim['m'], bsim['f'], sb['m'], sb['f'])
    print(msg, flush=True)
    f.write('# ' + msg + '\n'); f.flush()
    # listening clips: the first two paragraphs, base and best, both voices
    ld = a.out + '.listen'
    os.makedirs(ld, exist_ok=True)
    text = '\n'.join(open(os.path.join(ks.ROOT, 'kennedy.txt'), encoding='utf-8').read().split('\n')[:3])
    pf = os.path.join(ld, 'best.params')
    with open(pf, 'w') as o:
        for k in sorted(best['params']):
            o.write('%s %.9g\n' % (k, best['params'][k]))
    import subprocess
    for v in 'mf':
        for nm, env in (('base', {}), ('best', {'PUHUJA_PARAMS': pf})):
            e = dict(os.environ, **env)
            if not env:
                e.pop('PUHUJA_PARAMS', None)
            subprocess.run([a.bin, '-en'] + (['-female'] if v == 'f' else []) + ['-o', os.path.join(ld, '%s-%s.wav' % (nm, v)), text],
                           env=e, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
