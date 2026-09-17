# Data requirements

The pipeline needs three CSV files in `data/raw/`. They were inspected locally but are not committed because their redistribution licenses were not provided, and `AllNews.csv` contains full third-party article text.

## Expected files

### `Companies.csv`

Headerless UTF-8/UTF-8-BOM CSV with 11 rows and three columns:

| Position | Name | Description |
|---:|---|---|
| 1 | `CompanyID` | Numeric company identifier |
| 2 | `NameAr` | Arabic company name |
| 3 | `NameEn` | English company name |

### `AllNews.csv`

UTF-8/UTF-8-BOM CSV with a header and 10,526 article-company rows:

| Column | Description |
|---|---|
| `ArticleID` | Article identifier |
| `CompanyID` | Company identifier matching `Companies.csv` |
| `CompanyNameAr` | Arabic company name |
| `News` | Full Arabic article text |
| `PublishedOn` | Publication timestamp parsed by pandas using mixed formats |

The code removes duplicate `(ArticleID, CompanyID)` pairs.

### `StockPrices.csv`

Headerless UTF-8/UTF-8-BOM CSV with 13,726 rows and 14 columns:

```text
RowID, Date, MarketCode, CompanyID, Col4, Volume, Amount,
OpenPrice, ClosePrice, MinPrice, MaxPrice, Trades, F12, F13
```

The pipeline deduplicates `(CompanyID, Date)` and computes close-to-close, open-to-close, volume-change, lag, market-proxy, and forward-return variables.

## Local verification values

Use these SHA-256 values to confirm that you are rerunning with the exact reviewed inputs:

```text
c7bed5798095c8fbde1da803207a9d988e1db34441af9471d1610e3c464e2ff8  Companies.csv
094065713a281fb14b6166abd49645330d98ee9ad739c0d8874eb1a9c3801dda  AllNews.csv
793cb7793eb30cc6ca482715bdebf8bc54f4a077723950cced07bb476bb79625  StockPrices.csv
```

On macOS or Linux:

```bash
sha256sum data/raw/Companies.csv data/raw/AllNews.csv data/raw/StockPrices.csv
```

## Before distributing data

Confirm the original news and market-data providers, license terms, attribution requirements, and whether full-text redistribution is permitted. If it is not permitted, publish a data-access statement and a script that reconstructs the dataset from authorized sources instead of uploading the raw files.

