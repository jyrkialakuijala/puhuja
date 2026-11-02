CXX ?= g++
CXXFLAGS ?= -O2 -march=native -std=c++17 -Wall -Wextra

puhuja: main.cc puhuja.cc puhuja.h filters.h params.h larynx.cc larynx.h grid.cc grid.h block.cc block.h string.cc string.h folds.cc folds.h line.cc line.h tract.cc tract.h english.cc english.h english_data.inc english_shape.inc
	$(CXX) $(CXXFLAGS) -o $@.new main.cc puhuja.cc larynx.cc grid.cc block.cc string.cc folds.cc line.cc tract.cc english.cc && mv $@.new $@  # atomic: a running search keeps the old binary

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
