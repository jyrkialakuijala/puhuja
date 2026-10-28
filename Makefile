CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

puhuja: main.cc puhuja.cc puhuja.h folds.cc folds.h english.cc english.h english_data.inc english_shape.inc
	$(CXX) $(CXXFLAGS) -o $@ main.cc puhuja.cc folds.cc english.cc

# Regenerate the English pronunciation tables from CMUdict.
english:
	python3 -I tools/english_build.py

clean:
	rm -f puhuja out.wav

test: puhuja
	python3 -I tools/test_english.py
