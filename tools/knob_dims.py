"""Pick the dimensions for a short knob-search run: the top share of knobs by their gains in earlier screens plus a
random share of the others (Jyrki 2026-11-20: "make a list of important dimensions (like top 5 % of the previous
searches) and add another 5 % of random directions, and optimize those with line search, then repeat").

  stt/venv/bin/python -I tools/knob_dims.py --bin BIN --out ref/opt/ksearch/run10 --screens run6,run8,run9 [--top 0.05]
      [--rand 0.05] [--seed 10] [--lam 60] [--base-params P]   -> <out>.only ("KNOB MULT" lines, both directions)

Eligible: knobs in the pool of BIN (knob_search.pool: no integer / threshold / per-voice knobs), not timing knobs (the
knob_line.py regex), not byte-identical to the base both ways in every screen that has them (dead).
Score of a knob in one screen: the better side of x0.9 / x1.1 of (m + f Whisper errors per seed) + lam * (m + f
similarity change) where that screen has a similarity row, 0 if both sides are worse; a knob's score is the mean over
the screens that have it (a knob seen in one screen only keeps that one, noisier, value). The top share by score (most
negative first) and a random share of the rest go to <out>.only, each with 0.9 and 1.1.
"""
import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knob_search as ks  # noqa: E402

TIMING = re.compile(r'LEN|DUR|TEMPO|PAUSE|_MS|RATE|SPEED|HOLD|TIMING|Length')
TRACT = re.compile(r'^LINE_V\d+_LEN$|^LINE_LARYNX_LEN$')


def screen_scores(prefix, lam):
    sim = {}
    if os.path.exists(prefix + '.sim.tsv'):
        for l in open(prefix + '.sim.tsv'):
            c = l.rstrip('\n').split('\t')
            if c[0] != 'name':
                sim[(c[1], round(float(c[2]), 3))] = float(c[5]) + float(c[6])
    side = {}
    dead = {}
    for l in open(prefix + '.screen.tsv'):
        c = l.rstrip('\n').split('\t')
        if c[0] in ('name', 'base'):
            continue
        k, m = c[1], round(float(c[2]), 3)
        ns = len(c[4].split(','))
        side.setdefault(k, {})[m] = (int(c[6]) + int(c[13])) / ns + lam * sim.get((k, m), 0.0)
        dead.setdefault(k, []).append(c[7] == 'same' and c[14] == 'same')
    sc = {k: min(0.0, min(d.values())) for k, d in side.items()}
    return sc, {k: all(v) for k, v in dead.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--screens', required=True, help='comma-separated screen prefixes (ref/opt/ksearch/runN or runN)')
    ap.add_argument('--top', type=float, default=0.05)
    ap.add_argument('--rand', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lam', type=float, default=60.0)
    a = ap.parse_args()
    P = ks.pool(a.bin, a.out)
    elig = {k for k in P if not (TIMING.search(k) and not TRACT.search(k))}
    tot = {}
    alive = {}
    for s in a.screens.split(','):
        pre = s if os.sep in s else os.path.join(ks.ROOT, 'ref/opt/ksearch', s)
        if not os.path.exists(pre + '.screen.tsv'):
            continue
        sc, dead = screen_scores(pre, a.lam)
        for k, v in sc.items():
            tot.setdefault(k, []).append(v)
        for k, d in dead.items():
            alive[k] = alive.get(k, False) or not d
    elig = {k for k in elig if alive.get(k, True)}
    ranked = sorted((k for k in elig if k in tot), key=lambda k: sum(tot[k]) / len(tot[k]))
    ntop = max(1, int(round(a.top * len(elig))))
    nrand = max(1, int(round(a.rand * len(elig))))
    top = [k for k in ranked[:ntop] if sum(tot[k]) < 0]
    rest = sorted(elig - set(top))
    rnd = random.Random(a.seed).sample(rest, min(nrand, len(rest)))
    with open(a.out + '.only', 'w') as f:
        for k in top + rnd:
            for m in (0.9, 1.1):
                f.write('%s %.1f\n' % (k, m))
    with open(a.out + '.dims.tsv', 'w') as f:
        f.write('knob\tkind\tscore\tnscreens\n')
        for k in top:
            f.write('%s\ttop\t%.2f\t%d\n' % (k, sum(tot[k]) / len(tot[k]), len(tot[k])))
        for k in rnd:
            f.write('%s\trandom\t%s\t%d\n' % (k, '%.2f' % (sum(tot[k]) / len(tot[k])) if k in tot else '-', len(tot.get(k, []))))
    print('%d eligible knobs; %d top + %d random -> %d candidates in %s' % (len(elig), len(top), len(rnd), 2 * (len(top) + len(rnd)),
                                                                          a.out + '.only'))


if __name__ == '__main__':
    main()
