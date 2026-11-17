# The knob search: finding better defaults one knob at a time

How puhuja's ~3900 knobs are searched with Whisper as the judge, what went wrong the first time, and how to run it.
Written 2026-11-17 after search run1 and its repair. The step-by-step record (every run, number and decision) is
ref/opt/ksearch/LOG.md; the owner's words are in INPUT.md.

## Why

On 2026-11-15 sixteen autonomous "aesthetic" deploys made the voice worse ("strange bass sounds", "like an AM radio
that is not tuned right"); one of them (VQCS: breath and shimmer in connected speech) was the culprit, and Whisper had
*improved* on it. On 2026-11-16 Jyrki asked for a different approach: "a search on trying to find better parameters
by +- 10 % or +- 5 % search on values and looking at whisper and other scores", then for combining the findings by
"go[ing] first linearly towards the gradient then dropping 75 % of the dimensions and going in that direction as
long as optimum improves, and again reducing 50 % of the dimensions in future steps".

## The tools

| tool | what it does |
|---|---|
| `params.h` `PUHUJA_PARAM_LOG=file` | logs every knob a render reads, with the default in effect: the pool of a text |
| `tools/knob_search.py screen` | each knob x0.9 and x1.1 from the base, nothing accepted: a slope per knob |
| `tools/knob_search.py confirm` | the best N knob-directions at x0.9 / 0.95 / 1.05 / 1.1 on fresh seeds |
| `tools/knob_search.py eval` / `base` | any params file (or the base) on given seeds |
| `tools/knob_line.py` | Jyrki's plan: a line search along the gradient in all knobs, then the steepest 25 %, then 50 % per stage; validation on fresh seeds; listening clips |
| `tools/deploy_gate2.sh` | the dictionary gate (14646 words x 4 draws) - every deploy passes it |
| `tuned_defaults.h` | defaults for knobs whose names are built at run time (`DUR_CS_P_<ph>`, `LINE_CAV_<n>_*`, ...) and per-voice defaults (`NAME@f` / `NAME@m`) |

`param()` order: a params file (`PUHUJA_PARAMS`, `NAME@f` / `NAME@m` lines apply to one voice) > the environment >
`tuned_defaults.h` (`NAME@f` / `NAME@m` first) > the default in the code. The voice comes from `-female` on the
command line (`paramVoiceFemale()`), because many knobs are read at static initialisation, before `main()`.

## The score of a candidate

For each voice and each seed (`PUHUJA_SEED`):

1. **Kennedy Whisper:** kennedy.txt rendered with `-words`, cut into its 41 sentences, each sentence heard by
   Whisper medium.en (one model kept loaded), word errors summed. One seed: male ~58, female ~52 errors, sd ~3.5.
2. **Guard words** (`--words ref/survey/guard-ksB.txt`, since 2026-11-17): the 466 dictionary words the first search
   lost, rendered one per line as the gate does and heard in one batched pass; their wrong words are added to the
   voice's errors and kept apart in `m_werr` / `f_werr`. Jyrki: "make a list of those 723 words that got worse ...
   and try to read them next time you are making a kennedy optimization, so that those words will not get worse".
3. **Sound checks** against the base on the same seed: level (active-frame RMS), peak, bangs (5 ms frames 15 dB over
   their 105 ms median), bass (20-110 Hz share). Level and bass are steady to +-0.02 dB across seeds, so they flag
   any change of the voice itself; peaks and bangs move with the seed and only flag large changes. These exist
   because the VQCS regression improved Whisper while the sound got worse.

A render byte-identical to the base's costs no transcription (its knob is dead in that text).

## The pipeline

1. **Screen** (2 seeds per voice): every pool knob at x0.9 and x1.1. Run1: 4690 candidates, ~6.7 per minute on two
   GPU shards (~12 h); 979 dead. The m+f change spread p10 -8 / median +1 / p90 +11: the single leaders near -25
   are what chance gives among thousands of draws, so nothing is accepted here.
2. **Confirm** (8 fresh seeds per voice): the best 40 knob-directions at four step sizes. A knob counts if it wins
   again at both steps in the same direction. Run1 confirmed e.g. EN_LONG_PHRASE_TEMPO x1.1 -144, LINE_V4_LEN x1.1
   -138, PITCH_NATURAL_F0 x0.9 -86, LINE_LARYNX_LEN x0.9 -56, EN_DECLINATION x0.9 -55.
3. **Gradient:** per knob, the gain on the *better* side of x0.9 / x1.1 (not the central difference: a cliff on one
   side - LINE_Z_X x1.1 +134 errors - would read as a steep slope toward a value that is itself worse); worse both
   ways = 0 (the knob is at its optimum); confirm-stage slopes replace screen ones; dead knobs and knobs that moved
   level or bass > 0.3 dB in the screen are left out.
4. **Line search** (8 seeds per voice, 41-48): every knob moves against its slope, the steepest 10 % at alpha 1;
   alpha 0.25, 0.5, 1, 2, ... while the score improves and the sound checks pass; then the steepest 25 % of the knobs
   from the best point, then 50 % per stage down to 2. Run1: nearly all the gain was the full-gradient step
   (916 -> 672 m+f); the pruned stages added one small step.
5. **Validation** on 16 fresh seeds (51-58, 61-68), because choosing the best of many points on the same seeds
   flatters it. Run1's best: male 971 -> 611 (-37 %), female 892 -> 723 (-19 %).
6. **The owner's choice, then the gate.** Run1's best read 10 % slower (199 duration / tempo / pause knobs moved
   longer). Jyrki chose variant B: every non-timing knob at the best point, the timing knobs 10 % of the way
   ("mixing 10 % of the time knobs from the 'best, all knobs' and 90 % of the original value"). B: male 971 -> 767,
   female 892 -> 814, the reading 1 % longer. Deployed as deployed177.
7. **Dictionary gate** before anything is final. B lost 723 dictionary words (46701 vs 47424) - see below.

## What went wrong, and the rules that came out of it

**Switches hidden in knobs.** A +-1 % step means nothing to a knob that is really a switch, a table row or a
threshold - except that it flips it:

- *Integer knobs* (`static_cast<int>(param(...))`, `const int k = param(...)`, run-time `*_ROW` names): B moved 13 of
  them by ~1 % and the truncation flipped them: LINE_NF_MUR_ROW 16 -> 15 (word-final n heard as m: teen -> team,
  screen -> scream, stone -> storm), LINE_IHO_CTX 1 -> 0 (give -> gave, did -> dead), EN_ER_MED 3 -> 2 (Peru -> crew),
  LINE_N_REL_ROW, LINE_GD_VOICED_CTX, SH_OFF_SPAN, LIQ_FEMALE, CSW_FIRST, LINE_VQF_DIMS 511 -> 513, ... Restoring them
  (deployed178) won back 569 of the 723 words with the Kennedy gain kept.
- *Threshold knobs:* floats compared with a constant near their value: LINE_TH_SHAPE_CTX 2 -> 1.99 fails `>= 2`
  (initial θ lost its tongue key: Thursday -> birthday); LINE_JH_PRESS 1 -> 0.99 passes `< 1` (a ʤ path switched on).
- *Guard knobs:* a knob at exactly 1.0 tested with `!= 1.0f` switches a whole code block on when moved to 0.997:
  LINE_IHO_LATE_AC (the stressed ɪ opened toward e), and 15 timing / transition knobs.

`knob_search.py` now leaves all of these out of the pool (`int_knobs()`, `threshold_knobs()`: 2345 -> 1954 knobs
for the Kennedy text). Found by the round AH agents ONSET and VOWDICT.

Two more pool exclusions (2026-11-17 evening): inline guards `param("X", d) != c` count as threshold knobs too
(DUR_CS_FUNC, NAS_SIN_QX_CS, GAP_*, SH_OFF_*: the first filter only saw `kVar != c`), and **knobs with a per-voice
default** (`NAME@f` / `NAME@m` in tuned_defaults.h, 18 on deployed182) are left out, because a candidate writes one
plain `NAME value` line and that overrides both voices (a step on DUR_CS_FUNC would undo the female restore of
deployed181). The `.knobs` cache marks them `voice`; a cache written before this has no marks, so delete it when the
base has per-voice defaults. When a result found on an older base moves to a base with per-voice defaults, write
those knobs as `NAME@m` (or `@f`) lines (ref/opt/ksearch/run5-port.sh does this).

**Kennedy alone is not an objective.** A search on one text optimises that text; knobs that also act on single words
can lose the dictionary. Hence the guard words in every later search, and the full gate before every deploy.

**Not every accidental switch was bad.** Two of B's flips help connected speech: LINE_IHO_LATE_AC (restoring it
everywhere cost the male +35 Kennedy errors) and the 15 timing switches (the male lost 24 when restored). They were
restored only where they hurt: the ɪ switch in one-word phrases (LINE_IHO_LATE_AC_W, deployed180), the timing
switches for the female only (`NAME@f` in tuned_defaults.h, deployed181).

**Other lessons:**

- Speaking slower pleases Whisper (the duration cheat): keep timing knobs apart and measure the reading's length.
- Knob names lie: a "timing" regex (LEN, _MS, ...) also caught vocal-tract lengths (LINE_V4_LEN, LINE_LARYNX_LEN).
  With those at the full step the female's Kennedy errors rose 892 -> 939 (variant B2).
- PITCH_NATURAL_F0 transposes the voice (the pitch controller never learns): B moved it 126 -> 118 (~1 st up). The
  male's Kennedy likes it; restoring the female's cost the gate 48 words. Pitch level stays the owner's call.
- The standard Kennedy seeds favoured the old base by luck (placebo re-scores ~1000 vs the base's 957); compare
  with re-scores of a change too small to hear, not with the base's one draw.
- 11labs `.words.tsv` timestamps are 0.1-0.3 s off per word; word spans for prosody come from `.align.tsv`.

## Where it stands (2026-11-17, deployed182 = ef4805c)

| | dictionary gate | Kennedy Whisper, 16 fresh seeds (m / f) |
|---|---|---|
| before the search (deployed175/176) | 47424 | 971 / 892 |
| variant B (deployed177) | 46701 | 767 / 814 |
| deployed182 | 47422 | B's gain kept (deployed178-182 left the male Kennedy reading byte-identical; the female gained 46 more on 8 seeds) |

Kennedy history entries 068-073 hold the before / after recordings.

## Running a search

```
cd /home/jyrki/puhuja; S=ref/opt/ksearch; cp puhuja $S/runN.bin        # freeze the base; never rebuild it mid-run
# screen (two GPU shards; --only FILE for a candidate list; --base-params FILE to search around a params file)
systemd-run --user --unit ksN-screen0 --nice=10 --working-directory=$PWD stt/venv/bin/python -I tools/knob_search.py \
  screen --out $S/runN --words ref/survey/guard-ksB.txt --seeds 0,11 --procs 14 --shard 0/2
#   (the same with --shard 1/2)
# confirm the best 40 on fresh seeds
stt/venv/bin/python -I tools/knob_search.py confirm --out $S/runN --top 40 --seeds 31,32,33,34,35,36,37,38 --words ref/survey/guard-ksB.txt
# line search + validation + clips
stt/venv/bin/python -I tools/knob_line.py --out $S/runN --words ref/survey/guard-ksB.txt
# gate the result before anything else
GATE_SRC_BIN=$S/runN.bin tools/deploy_gate2.sh ksN $S/runN.listen/best.params ref/opt/gate/<latest>-dict
```

Every stage appends to its tsv and skips finished candidates, so a crash or reboot loses only the one in flight.
Keep render processes under ~32 in total. Records: ref/opt/ksearch/LOG.md (append at every step), STATUS.txt and
snap/ (hourly, the ksearch-snap timer).

## Deploying a result

1. The full gate must not lose more than ~60 words (noise ~100); Kennedy and the guard words must not get worse in
   either voice; level / bass within 0.3 dB, no new bangs; the reading's length unchanged (no duration cheat).
2. Voice changes (pitch level, breath, timbre) and anything the owner can hear as character go to his ear first.
3. `tools/apply_params.py P puhuja.cc english.cc line.cc larynx.cc string.cc folds.h`; names it reports "not in the
   sources" go to tuned_defaults.h; build; the default render must be `cmp`-identical to the search binary with the
   params (Kennedy and the dict400 stream, both voices).
4. One commit per change (numbers in the message; no Co-Authored-By, no Claude), `ref/opt/blend/puhuja-deployedNNN`,
   a Kennedy history entry (`tools/kennedy_history.sh NAME BEFORE AFTER "desc" HASH`), a line in LOG.md and INPUT.md.
   Push only when the owner says "push".
