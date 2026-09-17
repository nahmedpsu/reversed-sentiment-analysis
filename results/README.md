# Results

`paper_outputs/` contains the canonical supplementary artifacts supplied with the revised results notebook:

- Tables 1–12 as CSV files;
- pooled equal-split and pooled ridge lexicons;
- `key_numbers.json` with manuscript-ready statistics;
- ten publication-quality PNG figures.

`results/generated/` is reserved for a fresh rerun and is ignored by Git. To compare a new run with the canonical outputs:

```bash
diff -ru results/paper_outputs results/generated
```

Small floating-point differences can occur across BLAS and library versions. Investigate differences in row counts, signs, material effect sizes, or file coverage rather than treating every final decimal as a failure.

## Key interpretation

- The same-session estimate is stronger than the predictive estimate, which is consistent with contemporaneous information and potential label circularity.
- Predictive 2022 pooled ridge performance is `r = 0.112`, balanced accuracy `0.545`, and AUC `0.550` on 1,521 firm-days.
- The combined control-plus-lexicon benchmark reaches `r = 0.124`, compared with `r = 0.081` for lag/volume/market controls alone.
- Removing market-wrap articles and explicit price-movement words reduces the same-session correlation to `0.112`.
- The trading exercise has negative annualized returns and becomes worse after modeled costs.

