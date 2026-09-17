# Contributing

1. Create a focused branch from `main`.
2. Do not commit raw news text, credentials, tokens, personal data, or licensed market data.
3. Keep the canonical outputs in `results/paper_outputs/` unchanged unless the analysis has been intentionally rerun and the manuscript is updated at the same time.
4. Run `python scripts/validate_repository.py` before committing.
5. Explain methodological changes, affected outputs, and expected numerical differences in the pull request.

For research changes, prefer one commit for code/method edits and a separate commit for regenerated outputs. This makes review easier.

