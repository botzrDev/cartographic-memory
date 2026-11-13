# Cartographic Memory — conformance harness

**There is no lossless summary, and Gauss proved it in 1827.**

Gauss's *Theorema Egregium* says a curved surface cannot be flattened without
distortion. There is no distortion-free map, and no distortion-free summary.
What you can do is account for where the distortion goes.

This repository is the free instrument from the paper
[Cartographic Memory: An Impossibility Framework and Distortion Ledger for
Agent Memory Consolidation](https://doi.org/10.5281/zenodo.24201763). It checks
three laws against your own text:

1. **Conservation.** Retained + dropped = total. The ledger must balance.
2. **Monotonicity.** Fidelity is non-increasing along a consolidation chain.
3. **Composition.** The distortion of a chain equals the distortion of the
   composite map. For linear maps the chained indicatrix is the ordered product
   of the per-step Jacobians.

The theorem is exact for surfaces with a metric. A corpus of episodes has no
canonical metric, which is why the transfer to memory is an analogy and not a
proof. This harness tests the analogy instead of assuming it.

## What is here

- `code/conformance.py` — the tf-idf instrument. Two experiments: a linear
  projection (where the laws are algebraic identities) and recursive extractive
  summarization on four real books.
- `data/` — Plato's *Republic*, Nietzsche's *Genealogy of Morals*,
  Machiavelli's *The Prince*, Hume's *Enquiry*. Public domain, Project
  Gutenberg.
- `results/results.json` — the output this repository produces. Run it and
  compare.

It takes under 30 seconds, needs no API key, makes no network call, and uses no
randomness.

## Run it

```bash
bash install.sh
source .venv/bin/activate
cd code && python conformance.py
```

`install.sh` creates `.venv/` in the repository root. Nothing is written to your
system Python. Requires Python 3.9 or newer.

You should see:

```
[Experiment A - linear projection, EXACT regime]
  Law 1 (conservation, per-pass & per-region): PASS  (12 checks)
  Law 2 (monotonicity):                       PASS  (4 books)
  Law 3 (composition, Jacobian product):       PASS  (4 books)
  Attestation (recomputed, tamper caught):     PASS  (4 books)

[Experiment B - recursive extractive summarization, real books]
  Law 1 (weight conservation):        PASS  (4 checks)
  Law 2 (fidelity monotone):          PASS  (4 books)
  Corruption A (truncation) booked:   PASS
  Corruption B (trivialization) booked:PASS
  Attestation: tamper detected:       PASS

  Fidelity decay (centroid cosine vs original), 3 recursive passes @50%:
    Plato-Republic         -> 0.9919  0.9743  0.9578
    Nietzsche-Genealogy    -> 0.9770  0.9404  0.8994
    Machiavelli-Prince     -> 0.9757  0.9308  0.8587
    Hume-Enquiry           -> 0.9809  0.9533  0.9109

OVERALL: ALL LAWS HOLD - evidence captured.
```

Values move a little with machine and scikit-learn version. The verdicts do not.

## What the four books show

| Regime | Conservation | Monotonicity | Composition |
|---|---|---|---|
| Linear projection (PCA) | exact | exact | exact |
| Extractive selection, 3 passes @50% | exact | holds on 4/4 | not tested here |

Fidelity falls every pass on every book. Machiavelli decays fastest, from 0.9757
to 0.8587 across three 50 percent passes. Plato decays slowest, 0.9919 to
0.9578. That difference is the interesting part: what a summarizer loses depends
on the book, not only on the compression ratio.

## The limit you should know about

The laws are **proven exact only in the linear regime**, where every assertion is
an algebraic identity. The linear experiment is a conformance test: it verifies
the ledger's arithmetic, not a claim about summarizers.

The extractive experiment is the one genuinely empirical result here, and its
monotonicity is not forced by the code. Nothing makes a subset's centroid move
away from the original's; it just does.

The paper tests a third regime, LLM-based abstractive summarization, on eight
books. There, global monotonicity holds but per-region monotonicity fails in six
of eight, and chaining two passes always loses more fidelity than one deeper
pass. A sentence-embedding robustness check then shows the results are
instrument-dependent: Hume's fidelity *rises* across passes under embeddings,
and per-region violations jump from 11 to 43 of 128 comparisons.

**Instrument dependence is a finding, not a bug.** Different similarity functions
expose different failure modes, and a per-region ledger catches what a global
number hides.

## Citation

```bibtex
@misc{green2026cartographic,
  author    = {Green, Phillip Austin},
  title     = {Cartographic Memory: An Impossibility Framework and Distortion
               Ledger for Agent Memory Consolidation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24201763},
  url       = {https://doi.org/10.5281/zenodo.24201763}
}
```

## The full kit

This repository is the free instrument and it is complete for what it claims.

The paid kit on Gumroad adds the parts that make the rest of the paper
reproducible: the same instrument extended to twelve books, the sentence-embedding
instrument with its pre-computed results (the run that produces the
instrument-dependence finding), the bundled paper, one-command setup, 48-hour
support, and updates.

## Licence

Code: MIT, see [LICENSE](LICENSE). Use it in your own systems without
restriction. The paper and documentation are CC-BY-4.0. Cite the paper if you
publish results.

## Author

Phillip Austin Green — Botzr Research Labs, Oklahoma City
austin@botzr.com · [botzr.com](https://botzr.com) · [@botzr_research](https://x.com/botzr_research)
