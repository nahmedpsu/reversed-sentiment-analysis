# Notebook and artifact audit

Audit date: 2026-09-16.

## Files reviewed

- Revised results notebook: 33 cells (17 markdown, 16 code), all code cells marked executed.
- Earlier multi-company notebook: 32 cells (14 markdown, 18 code), all code cells marked executed.
- Final supplementary archive: 25 files (tables, lexicons, figures, and key numbers).
- Earlier analysis archive: three raw inputs plus firm-level lexicons, test scores, summaries, and the legacy notebook.

Checkpoint duplicates were intentionally excluded from the repository.

## Strengths

- Clear training/test split and a fixed random seed.
- Both contemporaneous and predictive news alignment are evaluated.
- Walk-forward, placebo, horizon, benchmark, ablation, transfer, and trading checks are present.
- Final figures and tables are already exported in portable formats.
- Saved output values are internally consistent between the revised notebook and supplementary archive.
- No API keys or embedded credentials were found in the notebook or source artifacts.

## Issues corrected for the repository

- Extracted the notebook pipeline into a command-line Python script.
- Added explicit `--data-dir` and `--output-dir` arguments and missing-input checks.
- Removed an unused logistic-regression import.
- Added environment definitions, validation automation, data schemas, checksums, citation metadata, licensing language, and a reproducibility guide.
- Kept the earlier notebook in an archive folder and made the revised notebook canonical.
- Excluded `.ipynb_checkpoints` and generated rerun folders.

## Remaining cautions

1. **Raw-data rights:** full news text and market data should not be redistributed until the providers' terms are documented.
2. **Fresh rerun:** supplied inputs were available for local verification, but exact rerun equivalence depends on the recorded package/BLAS environment.
3. **Multiple testing:** the analysis reports many firms, horizons, folds, and variants; manuscript claims should distinguish primary tests from exploratory checks.
4. **Placebo resolution:** 50 permutations imply coarse tail resolution; a publication-grade robustness appendix should consider at least 1,000 permutations if runtime permits.
5. **Trading assumptions:** the cost model charges a fixed daily amount when any position is open rather than modeling position-level turnover and slippage.
6. **Causality:** these are predictive associations. They do not prove that news causes the price movement.
7. **Citation metadata:** confirm the exact manuscript title, all authors, DOI, and repository release before public publication.

## Clean rerun

The extracted command-line pipeline completed successfully against the reviewed inputs on 2026-09-16 using Python 3.12, NumPy 2.5.3, pandas 2.3.3, SciPy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, and Matplotlib 3.11.2.

All canonical CSV files had matching shapes, columns, labels, and row identities. Equal-split lexicon values were exact. Ridge coefficients differed by less than `9.3e-7` after sorting by word. Across the main result tables, the largest numeric difference was approximately `0.00113` in a Newey–West t statistic; effect directions and manuscript conclusions were unchanged. Figure files were visually inspected for readable labels, legends, axes, and unclipped content.
