# Protein → Music

Turn disease-causing protein mutations into sound — using ESM-2 embeddings to
measure how "unusual" a mutation looks to a model trained only on evolution.

## 🎵 Try it live

https://huggingface.co/spaces/H9kkk/protein-music

Search any human protein, pick a real documented disease variant, and take
the blind listening test — can you tell which audio track is the mutant,
just by ear?

## What it does

1. Pick any human protein (search via UniProt) and one of its real documented
   disease-causing variants
2. The app fetches both the healthy and mutant sequence
3. Both are run through ESM-2 (8M param, runs on CPU) to get per-residue embeddings
4. The embedding distance between healthy and mutant is computed at each position —
   this spikes sharply at the mutation site (e.g. p53 R175H: ~12x the local average)
5. That disruption signal is sonified — higher disruption = more dissonant — and
   rendered as a chart

## How it works — honesty section

- The **detection signal** (embedding distance) is genuine, unmanipulated output
  from ESM-2. No disease labels were used in training.
- The **sonification rules** (disruption → dissonance, pitch mapping, etc.) are
  hand-designed, not learned. This is "pretrained model as feature extractor +
  deterministic mapping," not novel ML.
- This is an exploratory/educational tool, not a diagnostic one.

## Code overview

The Python files in this repo implement the pipeline:

| File | Role |
|---|---|
| `fetcher.py` | UniProt REST API — sequences, variants, mutations |
| `embedder.py` | ESM-2 embeddings, distance computation, music parameter mapping |
| `composer.py` | Converts music parameters into note sequences |
| `synth.py` | Renders notes to `.wav` directly (sine waves + harmonics) |
| `server.py` | Flask app — routes, ties everything together |
| `templates/index.html` | Frontend — search, charts (Chart.js), audio players |

## Tech stack

- **Model**: facebook/esm2_t6_8M_UR50D (Hugging Face transformers)
- **Backend**: Flask
- **Data**: UniProt REST API
- **Audio**: numpy + scipy (direct waveform synthesis, no soundfont)
- **Charts**: Chart.js
- **Hosting**: Hugging Face Spaces (Docker)

## Ideas for extension

- Train a classifier on ESM-2 deltas + ClinVar labels to predict pathogenicity
  (would make this a genuine ML contribution beyond feature extraction)
- 3D structure visualization (AlphaFold) showing mutation site in context
- Crowd-sourced blind test results — track aggregate "% correctly identified"
