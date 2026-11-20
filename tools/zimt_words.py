"""Word-by-word (or syllable-by-syllable) Zimtohrli distance between puhuja and
readings of the same text.
  tools/venv/bin/python -I tools/zimt_words.py kennedy.txt ref/kennedy/koko-0006-m.wav [more.wav ...]
      [--manifest ref/kokoro/kennedy-voices.txt --sex m] [--female] [--bin ./puhuja]
      [--syllables] [--best 10] [--baseline 6] [--worst 25] [--out words.tsv] [-- puhuja options]
Each reading needs <name>.words.tsv (start, end, word) next to its wav: from
tools/transcribe.py (Whisper) or tools/kokoro_voices.py (the pipeline's own
timestamps). puhuja reads the text with -words, which gives its word spans.
The readings' words are matched to the text's by spelling (difflib), so a
reading may cover more text or miss a word.
For every word, puhuja's cut and each reading's cut (10 ms padding, scaled
to the same RMS) go through Zimtohrli (Google's psychoacoustic similarity
metric at 48 kHz; it time-warps the spectrograms itself, so different
durations compare). With several readings the word's score is the weighted
mean of the --best closest readings, weights linear from 1 (closest) to
0.01 (the --best-th), so the voices a word resembles count and the rest
hardly do. The same score is computed for --baseline of the readings against
the other readings (leave-one-out), which says how a human reading scores
on the same word; the report lists the words where puhuja is farthest
beyond that baseline (ratio), the medians, and writes --out (per word).
--syllables splits puhuja's words at its syllable onsets (phoneme labels)
and each reading's word at the same relative times (approximate).
"""
import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
try:
    from scipy.io import wavfile
except ImportError:  # knob_line.py imports this module from stt/venv (no scipy); it reads wavs itself
    wavfile = None
import zimtohrli

HERE = os.path.dirname(os.path.abspath(__file__))
VOWELS = set("aeiouyæøəɪɛʊɔʌɑɒɜ")


def read_wav(path):
    fs, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if np.abs(x).max() > 2:
        x /= 32768.0
    if fs != 48000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(48000, fs)
        x = resample_poly(x, 48000 // g, fs // g).astype(np.float32)
    return x


def read_tsv(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3:
            rows.append((float(f[0]), float(f[1]), f[2]))
    return rows


def key(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def match_words(text_words, rows):
    """text index -> row index, by spelling."""
    a = [key(w) for w in text_words]
    b = [key(r[2]) for r in rows]
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
        elif tag == "replace" and i2 - i1 == j2 - j1:  # same count, different spelling (numbers, hyphens)
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
    return m


def trim(s, db=-30.0, frame=240):
    """Drop leading and trailing silence: 5 ms frames whose RMS is more than
    db below the loudest frame (Jyrki 2026-11-07: "trimming the silent sound in
    the beginning and end from the voices might be a good idea before feeding
    to zimtohrli")."""
    n = len(s) // frame
    if n < 2:
        return s
    e = np.sqrt(np.mean(s[:n * frame].reshape(n, frame) ** 2, axis=1))
    on = np.nonzero(e > e.max() * 10 ** (db / 20))[0]
    if len(on) == 0:
        return s
    return s[on[0] * frame:min(len(s), (on[-1] + 1) * frame)]


def cut(x, a, b, pad=0.01, db=-20.0, trim_db=None):
    i0 = max(0, int((a - pad) * 48000))
    i1 = min(len(x), int((b + pad) * 48000))
    s = x[i0:i1]
    if trim_db is not None and np.isfinite(s).all():
        s = trim(s, trim_db)
    if len(s) < 48000 // 50:
        return None
    # Garbage audio (NaN from unstable physics, 2026-10-31): Zimtohrli's
    # distance on it allocates without bound and took the machine down three
    # times; the word is left unscored instead.
    if not np.isfinite(s).all():
        return None
    if db is None:  # keep the level: Zimtohrli then also judges loudness
        s = np.asarray(s, dtype=np.float32)
    else:
        rms = np.sqrt(np.mean(s ** 2)) + 1e-9
        s = (s / rms * 10 ** (db / 20)).astype(np.float32)
    # Zimtohrli corrupts memory on signals under ~75 ms: pad with silence,
    # centred, to at least 200 ms (both sides get the same treatment).
    floor = 48000 // 5
    if len(s) < floor:
        left = (floor - len(s)) // 2
        s = np.concatenate([np.zeros(left, np.float32), s, np.zeros(floor - len(s) - left, np.float32)])
    return s


def render(text_path, binary, female, extra, workdir):
    wav = os.path.join(workdir, "puhuja.wav")
    lab = os.path.join(workdir, "puhuja.lab")
    words = os.path.join(workdir, "puhuja.words")
    cmd = [binary, "-en", "-o", wav, "-l", lab, "-words", words] + (["-female"] if female else []) + extra
    with open(text_path, encoding="utf-8") as f:
        subprocess.run(cmd, stdin=f, check=True)
    return read_wav(wav), read_tsv(lab), read_tsv(words)


def syllables_of(lab, a, b):
    """Syllable spans in [a, b] from phoneme labels: a syllable starts at the
    consonant run before each vowel (the first run belongs to the first)."""
    segs = [(s, e, p) for s, e, p in lab if s >= a - 1e-4 and e <= b + 1e-4 and p != "_"]
    if not segs:
        return [(a, b)]
    isv = [p[0] in VOWELS for _, _, p in segs]
    starts = [segs[0][0]]
    i = 0
    seen_vowel = False
    while i < len(segs):
        if isv[i]:
            if seen_vowel and not isv[i - 1]:
                j = i - 1
                while j > 0 and not isv[j - 1]:
                    j -= 1
                starts.append(segs[j][0])
            seen_vowel = True
        i += 1
    starts = sorted(set(starts))
    spans = [(s, starts[k + 1] if k + 1 < len(starts) else segs[-1][1]) for k, s in enumerate(starts)]
    spans = [(s, e) for s, e in spans if e - s > 0.03]
    return spans or [(a, b)]


def weighted(ds, best):
    ds = np.sort(np.asarray([d for d in ds if np.isfinite(d)], float))
    if len(ds) == 0:
        return np.nan
    ds = ds[:best]
    w = np.linspace(1.0, 0.01, best)[: len(ds)] if len(ds) > 1 else np.ones(1)
    return float(np.sum(w * ds) / np.sum(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("refs", nargs="*")
    ap.add_argument("--manifest", help="tools/kokoro_voices.py's <prefix>-voices.txt: wav name, voice, sex")
    ap.add_argument("--sex", choices=["m", "f"], help="with --manifest: only these voices")
    ap.add_argument("--female", action="store_true")
    ap.add_argument("--bin", default=os.environ.get("PUHUJA_BIN") or os.path.join(HERE, "..", "puhuja"))
    ap.add_argument("--syllables", action="store_true")
    ap.add_argument("--best", type=int, default=10)
    ap.add_argument("--baseline", type=int, default=6, help="readings scored against the others, for the scale")
    ap.add_argument("--worst", type=int, default=25)
    ap.add_argument("--out")
    ap.add_argument("--as", dest="as_", help="score this reading (wav with .words.tsv) instead of puhuja: a human's distance to the pool")
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    args = ap.parse_args(argv)
    refs = list(args.refs)
    if args.as_ and args.syllables:
        sys.exit("--syllables needs puhuja's phoneme labels; not with --as")
    if args.manifest:
        d = os.path.dirname(args.manifest)
        for line in open(args.manifest, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and (not args.sex or f[2] == args.sex):
                refs.append(os.path.join(d, f[0]))
    if not refs:
        sys.exit("no readings given")
    text_words = open(args.text, encoding="utf-8").read().replace("–", " ").replace("—", " ").split()
    if args.as_:
        px = read_wav(args.as_)
        rows = read_tsv(os.path.splitext(args.as_)[0] + ".words.tsv")
        m = match_words(text_words, rows)
        pwords = [(rows[m[i]][0], rows[m[i]][1], text_words[i]) if i in m else (0.0, 0.0, text_words[i]) for i in range(len(text_words))]
        plab = []
        print(f"{args.as_}: {len(m)} of {len(text_words)} words matched", file=sys.stderr)
    else:
        with tempfile.TemporaryDirectory() as td:
            px, plab, pwords = render(args.text, args.bin, args.female, extra, td)
        if len(pwords) != len(text_words):
            print(f"puhuja gave {len(pwords)} words for {len(text_words)} in the text", file=sys.stderr)
    n = min(len(pwords), len(text_words))
    readings = []
    for r in refs:
        rows = read_tsv(os.path.splitext(r)[0] + ".words.tsv")
        m = match_words(text_words, rows)
        readings.append((os.path.basename(os.path.splitext(r)[0]), read_wav(r), rows, m))
        if len(m) < n * 0.9:
            print(f"{r}: only {len(m)} of {n} words matched", file=sys.stderr)
    z = zimtohrli.Pyohrli()
    # ZIMT_SMEAR=<channels>: Gaussian smear across frequency channels before the
    # comparison (Jyrki 2026-11-01: a formant-tracker-like distance; 2 ~ six channels).
    z.frequency_smear = float(os.environ.get("ZIMT_SMEAR", "0"))
    z.frequency_smear_hz = float(os.environ.get("ZIMT_SMEAR_HZ", "0"))  # or in Hz (wide at low frequencies)
    names = [r[0] for r in readings]
    R = len(readings)
    base_idx = list(range(min(args.baseline, R))) if R > 1 else []

    units = []  # (label, puhuja span, [reading span or None])
    for i in range(n):
        pa, pb, _ = pwords[i]
        rsp = []
        for _, _, rows, m in readings:
            j = m.get(i)
            rsp.append((rows[j][0], rows[j][1]) if j is not None else None)
        if not args.syllables:
            units.append((text_words[i], (pa, pb), rsp))
            continue
        syl = syllables_of(plab, pa, pb)
        for k, (sa, sb) in enumerate(syl):
            ra, rb = (sa - pa) / max(pb - pa, 1e-6), (sb - pa) / max(pb - pa, 1e-6)
            rs = []
            for sp in rsp:
                if sp is None:
                    rs.append(None)
                else:
                    a, b = sp
                    rs.append((a + ra * (b - a), a + rb * (b - a)))
            units.append((f"{text_words[i]}#{k + 1}/{len(syl)}", (sa, sb), rs))

    per_ref = np.full((len(units), R), np.nan)   # puhuja vs each reading
    score_p = np.full(len(units), np.nan)        # puhuja's weighted best
    score_b = np.full((len(units), len(base_idx)), np.nan)  # baseline readings' weighted best vs the others
    for u, (label, (pa, pb), rsp) in enumerate(units):
        ps = cut(px, pa, pb)
        cuts = [cut(x, *sp) if sp else None for (_, x, _, _), sp in zip(readings, rsp)]
        if ps is not None:
            per_ref[u] = [z.distance(ps, c) if c is not None else np.nan for c in cuts]
            score_p[u] = weighted(per_ref[u], args.best)
        for bi, b in enumerate(base_idx):
            if cuts[b] is None:
                continue
            ds = [z.distance(cuts[b], c) for j, c in enumerate(cuts) if j != b and c is not None]
            score_b[u, bi] = weighted(ds, args.best)
    base_u = np.nanmean(score_b, axis=1) if base_idx else np.full(len(units), np.nan)

    print(f"{'syllable' if args.syllables else 'word'}s: {len(units)}; readings: {R}; best {args.best} weighted 1 -> 0.01"
          f" (Zimtohrli distance, lower = more alike)")
    print(f"  puhuja:    median {np.nanmedian(score_p):.4f}  mean {np.nanmean(score_p):.4f}")
    for bi, b in enumerate(base_idx):
        print(f"  {names[b]:14s} median {np.nanmedian(score_b[:, bi]):.4f}  mean {np.nanmean(score_b[:, bi]):.4f}  (against the other readings)")
    if base_idx:
        ok = np.isfinite(score_p) & np.isfinite(base_u)
        ratio = score_p[ok] / np.maximum(base_u[ok], 1e-6)
        print(f"  puhuja / human baseline: median ratio {np.median(ratio):.2f}; puhuja farther on {np.mean(ratio > 1) * 100:.0f} % of the units")
    if R > 1:
        order = np.argsort(np.nanmedian(per_ref, axis=0))
        print("  closest readings to puhuja (median over units): " + ", ".join(f"{names[j]} {np.nanmedian(per_ref[:, j]):.4f}" for j in order[:5]))
    worst = np.argsort(-(score_p / np.maximum(base_u, 1e-6))) if base_idx else np.argsort(-score_p)
    print(f"\n  worst {args.worst}" + (" (puhuja / baseline)" if base_idx else "") + ":")
    for u in worst[: args.worst]:
        if not np.isfinite(score_p[u]):
            continue
        extra_s = f" / {base_u[u]:.4f} = {score_p[u] / max(base_u[u], 1e-6):.2f}" if base_idx and np.isfinite(base_u[u]) else ""
        print(f"    {units[u][0]:22s} {score_p[u]:.4f}{extra_s}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("unit\tpuhuja\tbaseline\t" + "\t".join(names) + "\n")
            for u, (label, _, _) in enumerate(units):
                f.write(f"{label}\t{score_p[u]:.5f}\t{base_u[u]:.5f}\t" + "\t".join(f"{v:.5f}" for v in per_ref[u]) + "\n")


if __name__ == "__main__":
    main()
