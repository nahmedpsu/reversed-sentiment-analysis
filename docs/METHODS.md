# Method summary

## Study design

The study uses Arabic financial-news articles and daily market data for 11 Saudi-listed firms over 2018–2022. The main models are trained through 2021 and evaluated out of sample in 2022. Walk-forward folds train on all years before each test year from 2019 through 2022.

## Text processing

The normalization step removes Arabic diacritics and tatweel, standardizes common alef/ya/ta-marbuta variants, replaces Arabic and Western numeric strings with `NUM`, removes punctuation, and splits on whitespace. No stemming or pretrained language model is used.

## Reversed-sentiment estimators

1. **Equal split:** each token in a daily document receives an equal share of that day's return. Token scores are averaged over training observations and filtered to at least three distinct days.
2. **Ridge TF-IDF:** daily token lists are transformed with sublinear TF-IDF; words must occur in at least five documents. Ridge regression uses `alpha = 10` in the main specification.

The lexicons are frozen before test scoring.

## Timing definitions

- **Contemporaneous:** news is assigned to the calendar day's market session.
- **Predictive:** news is assigned to the first session opening after publication. Pre-open news maps to the same date; intraday and post-close news map to the next trading day.

The predictive definition is the relevant design for claims about pre-session information.

## Outcomes and statistics

The primary continuous outcome is close-to-close log return. Open-to-close log return is used in robustness and trading analyses. Reported metrics include Pearson and Spearman correlations, Newey–West/HAC t and p values, directional accuracy, balanced accuracy, ROC AUC, majority-class accuracy, and Pesaran–Timmermann directional p values.

Additional analyses cover shuffled-label placebos, publication timing, return horizons, lag/volume/market benchmarks, incremental regressions, firm transfer, vocabulary/penalty ablations, and a long/cash trading exercise.

## Interpretation boundaries

The design establishes out-of-sample association and limited predictive content, not a causal news effect. Same-session estimates can reflect articles describing price movements. The trading exercise simplifies transaction costs and does not model turnover, slippage, liquidity, bid–ask spreads, or capacity in full.

