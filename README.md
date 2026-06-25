# Probatio

[![CI](https://github.com/d-leontiev/probatio-check/actions/workflows/ci.yml/badge.svg)](https://github.com/d-leontiev/probatio-check/actions/workflows/ci.yml)

*Confidential, fully-local citation checker for scientific manuscripts.*

Probatio verifies a manuscript's in-text citations against the cited reference PDFs you
supply: it parses every in-text citation, resolves it to the matching reference PDF,
retrieves the verbatim passage that the claim rests on, and has a local LLM judge whether
the manuscript's claim is actually supported. The key property is **fail-closed
confidentiality** — before doing anything, it asserts that every model and endpoint is
running locally (loopback, RFC 1918, or Tailscale); if anything looks like a cloud
endpoint it refuses to run. An embargoed or in-preparation manuscript never leaves the
machine, unlike cloud AI writing tools that silently send your draft to a remote API.

## Install

```bash
git clone https://github.com/d-leontiev/probatio-check.git && cd probatio-check
pip install -e '.[local]'
```

The `local` extra adds `paper-qa[local]` for local sentence-transformers embeddings so
retrieval uses no external API key.

## Usage

```bash
probatio-check --manuscript paper.pdf --refs ./refs [--out ./results] [--serve] [--port 8000]
```

| Flag | Meaning |
|---|---|
| `--manuscript` | The manuscript PDF under review |
| `--refs` | Folder of the cited reference PDFs |
| `--out` | Output directory for `citations.json` / `citations.md` (default: alongside the manuscript) |
| `--serve` | Launch the citation audit UI when done (click-to-source, human override) |
| `--port` | Port for the audit UI (default: 8000) |

Outputs written to `--out`:
- **`citations.json`** — machine-readable record of every citation: the resolved reference,
  the retrieved verbatim passage, the verdict, and confidence.
- **`citations.md`** — human-readable summary table of the same.

### Audit UI

`--serve` (or `probatio-check-ui --run`) opens a three-pane reviewing instrument:

- **Queue** (left) — every citation, problems-first, with filter presets
  (Problems / Unreviewed / All), status pills, section filter, and search.
- **Verdict & reasoning** (centre) — the claim, the model's verdict + confidence
  and rationale, the cited reference's metadata, and one-key human override/confirm.
- **Evidence** (right) — the highlighted source page, the verbatim passage(s) with
  retrieval scores, and the contextual summary.

It is keyboard-driven (`?` shows the full map): `j`/`k` move, `1`–`6` set a verdict,
`r` confirms, `n` jumps to the next unreviewed, `/` searches. Overrides, confirmations,
and notes persist to `citations.json`. Like the rest of probatio, the UI is fully local:
the served HTML/CSS/JS make no off-machine requests (enforced by a test).

## Local model setup

Probatio requires an ollama instance with a capable judge model. For a machine with a
modern GPU (or via an SSH tunnel to one):

```bash
ollama pull gemma4:31b            # citation judge
ollama pull embeddinggemma        # retrieval embeddings
```

That's all the configuration needed: these models on a local ollama at
`http://localhost:11434` are the **defaults**, so check mode runs out-of-the-box with no
`.env`. Copy `.env.example` to `.env` only to override them.

If ollama runs on a remote GPU box, SSH-tunnel the port first:

```bash
ssh -L 11434:localhost:11434 user@gpu-box
```

The tunnel makes the remote ollama appear at the default `http://localhost:11434`, so no
override is needed. To point at it directly instead, set
`PROBATIO_OLLAMA_API_BASE=http://gpu-box:11434` (routes both judge and embeddings there).

## Verdict legend

| Verdict | Meaning |
|---|---|
| `supported` | The retrieved passage directly backs the manuscript's claim |
| `partially` | The passage is related but only partially supports the claim |
| `overstated` | The claim goes further than what the passage says |
| `unsupported` | The passage does not support the claim |
| `not_found` | No passage in the reference PDF matched the cited claim |
| `not_a_claim` | The in-text citation is to a method/tool/dataset, not a factual claim |

Human overrides are saved to `citations.json` by the audit UI.

## How it works

1. **Parse** — extract every in-text citation and its surrounding claim from the manuscript PDF.
2. **Resolve** — match each citation key to its reference PDF in the `--refs` folder.
3. **Retrieve** — embed the claim and retrieve the top-`k` verbatim passages from the
   reference PDF using local sentence-transformers (no network call).
4. **Judge** — a local LLM (gemma) reads the claim + passages and returns a structured
   verdict with a confidence score.

The fail-closed guard (`assert_local_only`) runs before step 1 and aborts the entire run
if any model or API base is not on loopback / RFC 1918 / Tailscale / a single-label LAN
hostname.
