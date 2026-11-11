CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fno-math-errno -fno-signed-zeros -fno-trapping-math -ffp-contract=off -std=c++17 -Wall -Wextra  # no -ffast-math: it defines __FAST_MATH__, which lets glibc swap exp/log/pow in vectorized loops for libmvec, so an unrelated code change (an inert knob) changed the rounding and the chaotic folds made it a different waveform (review dsp.md, 2026-11-11); without it an inert edit is bit-identical, and -O3 is faster (1.34 vs 1.70 s)

puhuja: Makefile main.cc puhuja.cc puhuja.h prosody.cc prosody.h filters.h params.h larynx.cc larynx.h grid.cc grid.h block.cc block.h string.cc string.h folds.cc folds.h line.cc line.h tract.cc tract.h english.cc english.h english_data.inc english_shape.inc
	$(CXX) $(CXXFLAGS) -o $@.new main.cc puhuja.cc prosody.cc larynx.cc grid.cc block.cc string.cc folds.cc line.cc tract.cc english.cc && mv $@.new $@  # atomic: a running search keeps the old binary

# Transmission-line tract prototype (line.h): sustained vowels from area functions.
linevowel: linevowel.cc line.cc line.h tract.cc tract.h string.cc string.h folds.cc folds.h params.h
	$(CXX) $(CXXFLAGS) -o $@ linevowel.cc line.cc tract.cc string.cc folds.cc

# Fit the soft tract shapes of the English vowels on the line (ref/line/vowel_tract.txt).
linefit: linefit.cc line.cc line.h tract.cc tract.h string.cc string.h folds.cc folds.h params.h english_shape.inc
	$(CXX) $(CXXFLAGS) -O3 -march=native -o $@ linefit.cc line.cc tract.cc string.cc folds.cc

# Regenerate the English pronunciation tables from CMUdict.
english:
	python3 -I tools/english_build.py

clean:
	rm -f puhuja out.wav

test: puhuja
	python3 -I tools/test_english.py
	@$(MAKE) -s chunktest

# Chunk invariance: Kennedy's last two paragraphs read as whole lines and fed 7 bytes at a time must be the same
# samples, both voices, with no render-time state in the plan (PLAN_INVARIANT 1, the breath planner's air scale fixed
# at the 0.5 it learned). It caught the learned scale (pauses 45-255 ms apart) and the reader holding a sentence's last
# word back from the breath planner (review cpp.md B2, 2026-11-11). PUHUJA_PARAMS=/dev/null: no puhuja.params.
CHUNKTEST = env PUHUJA_PARAMS=/dev/null PLAN_INVARIANT=1 SYNTH_DRAIN_SCALE=0.5 ./puhuja -en -r
chunktest: puhuja
	@a=$$(sed -n 13,15p kennedy.txt | $(CHUNKTEST) | md5sum) && b=$$(sed -n 13,15p kennedy.txt | $(CHUNKTEST) -chunk 7 | md5sum) && \
	 c=$$(sed -n 13,15p kennedy.txt | $(CHUNKTEST) -female | md5sum) && d=$$(sed -n 13,15p kennedy.txt | $(CHUNKTEST) -female -chunk 7 | md5sum) && \
	 if [ "$$a" = "$$b" ] && [ "$$c" = "$$d" ]; then echo "chunk invariance: ok"; else echo "chunk invariance: FAILED (male $$a / $$b, female $$c / $$d)"; exit 1; fi

# How close the string folds' glottal airflow is to the LF model in relaxed speech (a report).
lftest: puhuja
	python3 -I tools/test_lf.py

# Snappiness of the transients at the places in tests/snaps.tsv (tools/snap.py; Jyrki 2026-11-10).
snaptest: puhuja
	tools/venv/bin/python -I tools/snap.py --check
