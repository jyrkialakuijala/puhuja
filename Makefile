CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

puhuja: main.cc puhuja.cc puhuja.h filters.h params.h larynx.cc larynx.h grid.cc grid.h folds.cc folds.h english.cc english.h english_data.inc english_shape.inc
	$(CXX) $(CXXFLAGS) -o $@.new main.cc puhuja.cc larynx.cc grid.cc folds.cc english.cc && mv $@.new $@  # atomic: a running search keeps the old binary

# Regenerate the English pronunciation tables from CMUdict.
english:
	python3 -I tools/english_build.py

clean:
	rm -f puhuja out.wav

test: puhuja
	python3 -I tools/test_english.py
