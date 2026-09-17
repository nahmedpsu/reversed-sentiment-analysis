# Reproducibility checklist

## 1. Confirm data authorization

Obtain the three inputs from their lawful source and verify that your use complies with the applicable terms. Do not redistribute full article text unless the provider permits it.

## 2. Verify inputs

Place the CSVs in `data/raw/` and compare their SHA-256 checksums with [`data/README.md`](../data/README.md). Expected reviewed coverage is:

- 11 firms;
- 10,526 article-company rows;
- 13,726 price rows;
- dates spanning 2018–2022.

## 3. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda users can instead run:

```bash
conda env create -f environment.yml
conda activate reversed-sentiment
```

## 4. Run the analysis

```bash
python src/generate_paper_results.py \
  --data-dir data/raw \
  --output-dir results/generated
```

The script should generate 15 CSV/JSON/lexicon artifacts, two NumPy placebo arrays, and ten PNG figures. Runtime was approximately four minutes in the notebook's original environment.

## 5. Compare the rerun

Check the following anchors first:

| Anchor | Expected value |
|---|---:|
| Deduplicated article-company rows | 10,526 |
| Total predictive news days | 7,024 |
| Predictive 2022 test firm-days | 1,521 |
| Predictive pooled-ridge Pearson r | 0.112 |
| Predictive pooled-ridge Newey–West t | 4.463 |
| Predictive pooled-ridge balanced accuracy | 0.545 |
| Same-session pooled-ridge Pearson r | 0.326 |

Then compare the full files:

```bash
diff -ru results/paper_outputs results/generated
```

## 6. Notebook execution

From `notebooks/`, ensure the data path is `../data/raw` and the output path is `../results/generated`, then restart the kernel and run all cells. Clear transient warnings before publication, but retain the output tables that support the manuscript.

## 7. Archive a release

Before a public release:

1. replace placeholder manuscript metadata in `CITATION.cff`;
2. confirm data and derived-lexicon licensing;
3. add the accepted manuscript DOI, if available;
4. tag a semantic version such as `v1.0.0`;
5. archive the release with Zenodo and add the DOI badge to the README.
