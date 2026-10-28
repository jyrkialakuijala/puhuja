# puhuja

A streaming formant speech synthesizer with a physical vocal-fold source,
written in C++17. It reads Finnish (or IPA) and English text and writes
mono float32 audio at 48 kHz.

- Voice source: a two-mass model of the vocal folds (or simpler glottal,
  triangle and square carriers).
- Vocal tract: a cascade of formant resonators with a nasal pole/zero pair,
  turbulence noise at the constriction and breath noise at the glottis.
- English: on-the-fly grapheme-to-phoneme conversion (a compressed table
  built from CMUdict plus letter rules), text normalisation (numbers,
  currency, dates, abbreviations) and rule-based prosody.
- Streaming: text can be fed in pieces; audio comes out as it is ready.

## Build

    make

## Use

    ./puhuja -en -o hello.wav "Hello, how are you?"
    ./puhuja -en -r "Hello." | aplay -f FLOAT_LE -r 48000 -c 1
    ./puhuja "Hyvää huomenta."
    ./puhuja -en -o kennedy.wav < kennedy.txt

`./puhuja -h` lists the options (female voice, pitch, rate, voice source,
phoneme timings).

## License

Apache License 2.0; see `LICENSE`.

## Credits

The English pronunciation table (`english_data.inc`) is derived from the
CMU Pronouncing Dictionary, Copyright (C) 1993-2015 Carnegie Mellon
University; see `LICENSE-CMUDICT`.
