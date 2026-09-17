"""Generate all tables, figures, lexicons, and key numbers for the paper.

Run from the repository root:

    python src/generate_paper_results.py \
        --data-dir data/raw \
        --output-dir results/generated
"""
import argparse
import json
import os
from pathlib import Path
import re
import warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data/raw"),
    help="Folder containing Companies.csv, AllNews.csv, and StockPrices.csv.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("results/generated"),
    help="Folder for generated CSV, PNG, NPY, and JSON outputs.",
)
args = parser.parse_args()

D = args.data_dir
OUT = args.output_dir
required_inputs = [D / "Companies.csv", D / "AllNews.csv", D / "StockPrices.csv"]
missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
if missing_inputs:
    parser.error("Missing required input file(s): " + ", ".join(missing_inputs))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0; rng = np.random.default_rng(SEED)
TRAIN_END = pd.Timestamp("2021-12-31")
SESSION_OPEN_H = 10          # Tadawul session opens 10:00 (2018-2022)
SESSION_CLOSE_H = 15         # closes 15:00
MIN_DF = 5                   # regression vocabulary threshold (documents)
MIN_DAYS_EQ = 3              # equal-split lexicon threshold (distinct days)
ALPHA = 10.0                 # ridge penalty
R = {}                       # key numbers for the manuscript

# ------------------------------------------------------------------ data
companies = pd.read_csv(f"{D}/Companies.csv", header=None, names=["CompanyID", "NameAr", "NameEn"])
CIDS = companies.CompanyID.tolist(); NAME = companies.set_index("CompanyID").NameEn.to_dict()
PRICE_COLS = ["RowID", "Date", "MarketCode", "CompanyID", "Col4", "Volume", "Amount",
              "OpenPrice", "ClosePrice", "MinPrice", "MaxPrice", "Trades", "F12", "F13"]
prices = pd.read_csv(f"{D}/StockPrices.csv", header=None, names=PRICE_COLS, encoding="utf-8-sig")
prices["Date"] = pd.to_datetime(prices.Date).dt.normalize()
prices = prices.sort_values(["CompanyID", "Date"]).drop_duplicates(["CompanyID", "Date"]).reset_index(drop=True)
g = prices.groupby("CompanyID")
prices["cc"] = np.log(prices.ClosePrice / g.ClosePrice.shift(1))
prices["oc"] = np.log(prices.ClosePrice / prices.OpenPrice)
prices["dlv"] = np.log(prices.Volume) - np.log(g.Volume.shift(1))
for k in range(1, 6):
    prices[f"cc_l{k}"] = g.cc.shift(k)
prices["dlv_l1"] = g.dlv.shift(1)
mkt = prices.groupby("Date").cc.mean().rename("mkt")          # equal-weighted proxy of the 11 firms
prices = prices.merge(mkt, on="Date", how="left"); prices["mkt_l1"] = prices.groupby("CompanyID").mkt.shift(1)
H = 10
for k in range(0, H + 1): prices[f"f{k}"] = prices.groupby("CompanyID").cc.shift(-k)

news = pd.read_csv(f"{D}/AllNews.csv", encoding="utf-8-sig")
news["PublishedOn"] = pd.to_datetime(news.PublishedOn, format="mixed")
news["News"] = news.News.astype(str).str.replace("&nbsp;", " ", regex=False)
news = news.drop_duplicates(["ArticleID", "CompanyID"]).reset_index(drop=True)
hr = news.PublishedOn.dt.hour
news["timing"] = np.where(hr < SESSION_OPEN_H, "pre-open", np.where(hr < SESSION_CLOSE_H, "intraday", "post-close"))

# ------------------------------------------------------------------ text
_dia = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_pun = re.compile(r"[^\w\s]|_", flags=re.UNICODE)
_dig = re.compile(r"[0-9\u0660-\u0669]+([.,][0-9\u0660-\u0669]+)?")
def norm(t):
    t = _dia.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    t = _dig.sub(" NUM ", t); t = _pun.sub(" ", t); return t
def tok(t): return norm(t).split()
MOVE = set("ارتفع ارتفعت يرتفع ترتفع الارتفاع ارتفاع ارتفاعا مرتفعا مرتفعه تراجع تراجعت يتراجع تتراجع التراجع تراجعا متراجعا "
           "انخفض انخفضت ينخفض الانخفاض انخفاض انخفاضا منخفضا صعد صعود هبط هبوط مكاسب مكاسبه خسائر خسائره اعلي الاعلي "
           "ادني الادني اغلق اغلقت الاغلاق مغلقا ربح خسر قفز قفزت تهاوي".split())

# ------------------------------------------------------------------ alignment
def assign_session(n, tdays, mode):
    ts = n.PublishedOn
    if mode == "contemporaneous":
        tgt = ts.dt.normalize()
    else:  # "predictive": first session that opens after publication
        so = ts.dt.normalize() + pd.Timedelta(hours=SESSION_OPEN_H)
        tgt = pd.Series(pd.to_datetime(np.where(ts >= so, ts.dt.normalize() + pd.Timedelta(days=1), ts.dt.normalize())), index=n.index)
    td = np.asarray(tdays.values, dtype="datetime64[ns]")
    pos = np.searchsorted(td, tgt.values.astype("datetime64[ns]"), side="left"); ok = pos < len(td)
    out = n[ok].copy(); out["Date"] = pd.to_datetime(td[pos[ok]]); return out

def documents(cid, mode, drop_move=False, drop_wrap=False):
    p = prices[prices.CompanyID == cid].sort_values("Date")
    n = news[news.CompanyID == cid]
    if drop_wrap:
        n = n[~n.News.str.strip().str.startswith(("السوق السعودي", "تذكير", "مؤشر أرقام", "مؤشر ارقام"))]
    n = assign_session(n, pd.DatetimeIndex(p.Date), mode)
    d = (n.groupby("Date").agg(News=("News", " ".join), NumArticles=("ArticleID", "size")).reset_index()
           .merge(p.drop(columns=[c for c in p.columns if c == "News"]), on="Date"))
    d["Tokens"] = d.News.apply(tok)
    if drop_move: d["Tokens"] = d.Tokens.apply(lambda t: [w for w in t if w not in MOVE])
    d["CompanyID"] = cid
    return d.dropna(subset=["cc"]).sort_values("Date").reset_index(drop=True)

DOCS = {m: {c: documents(c, m) for c in CIDS} for m in ["contemporaneous", "predictive"]}
def pool(mode, cond): return pd.concat([DOCS[mode][c][cond(DOCS[mode][c])] for c in CIDS]).reset_index(drop=True)
is_train = lambda d: d.Date <= TRAIN_END
is_test = lambda d: d.Date > TRAIN_END

# ------------------------------------------------------------------ estimators
def eqsplit_lexicon(train, ycol="cc"):
    ex = train[["Date", "Tokens", ycol]].copy(); ex["M"] = ex.Tokens.str.len(); ex = ex[ex.M > 0]
    ex["S"] = ex[ycol] / ex.M; ex = ex.explode("Tokens").rename(columns={"Tokens": "Word"})
    lex = ex.groupby("Word").agg(Score=("S", "mean"), Freq=("S", "size"), DayFreq=("Date", "nunique")).reset_index()
    lex = lex[lex.DayFreq >= MIN_DAYS_EQ].copy()
    lex["Positive"] = lex.Score.clip(lower=0); lex["Negative"] = lex.Score.clip(upper=0)
    return lex.sort_values("Score", ascending=False).reset_index(drop=True)
def eqsplit_score(docs, lex):
    idx = dict(zip(lex.Word, lex.Score)); return np.array([sum(idx.get(w, 0.0) for w in t) for t in docs.Tokens])
def ridge_fit(train, ycol="cc"):
    vec = TfidfVectorizer(analyzer=lambda x: x, min_df=MIN_DF, sublinear_tf=True)
    X = vec.fit_transform(train.Tokens); return vec, Ridge(alpha=ALPHA).fit(X, train[ycol])
def ridge_score(docs, model): vec, rd = model; return rd.predict(vec.transform(docs.Tokens))

# ------------------------------------------------------------------ metrics
def nw(s, y, lags=1):
    s, y = np.asarray(s, float), np.asarray(y, float); ok = ~np.isnan(y) & ~np.isnan(s)
    X = sm.add_constant(s[ok]); f = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    return f.tvalues[1], f.pvalues[1]
def evaluate(s, y, lags=1):
    s, y = np.asarray(s, float), np.asarray(y, float); ok = ~np.isnan(y) & ~np.isnan(s); s, y = s[ok], y[ok]
    out = {"n": int(len(y))}
    if len(y) < 10 or s.std() == 0: return out
    out["r"] = stats.pearsonr(s, y)[0]; out["nw_t"], out["nw_p"] = nw(s, y, lags)
    out["spearman"] = stats.spearmanr(s, y)[0]
    ys = np.where(y > 0, 1, -1); ss = np.where(s > 0, 1, -1)
    out["acc"] = (ss == ys).mean()
    up, dn = ys == 1, ys == -1
    out["bal_acc"] = np.mean([(ss[up] == 1).mean() if up.any() else np.nan, (ss[dn] == -1).mean() if dn.any() else np.nan])
    out["auc"] = roc_auc_score((y > 0).astype(int), s) if 0 < (y > 0).mean() < 1 else np.nan
    out["majority"] = max((y > 0).mean(), (y <= 0).mean())
    n = len(y); py = (ys == 1).mean(); ps = (ss == 1).mean(); pstar = py * ps + (1 - py) * (1 - ps)
    v_p = pstar * (1 - pstar) / n
    v_s = ((2 * py - 1) ** 2 * ps * (1 - ps) + (2 * ps - 1) ** 2 * py * (1 - py) + 4 * py * ps * (1 - py) * (1 - ps) / n) / n
    out["pt_p"] = float(1 - stats.norm.cdf((out["acc"] - pstar) / np.sqrt(v_p - v_s))) if v_p > v_s else np.nan
    return out
def fmt(x, d=3): return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else (f"{x:.{d}f}" if isinstance(x, (float, np.floating)) else str(x))


rows = []
for c in CIDS:
    p = prices[prices.CompanyID == c]; n = news[news.CompanyID == c]
    dpre = DOCS["predictive"][c]
    rows.append({"CompanyID": c, "Company": NAME[c], "TradingDays": len(p), "Articles": len(n),
                 "PreOpen%": 100 * (n.timing == "pre-open").mean(), "Intraday%": 100 * (n.timing == "intraday").mean(),
                 "PostClose%": 100 * (n.timing == "post-close").mean(),
                 "NewsDays": len(dpre), "NewsDaysTrain": int(is_train(dpre).sum()), "NewsDaysTest": int(is_test(dpre).sum()),
                 "MeanWordsPerArticle": n.News.str.split().str.len().mean()})
t1 = pd.DataFrame(rows)
tot = {"CompanyID": "", "Company": "Total / mean", "TradingDays": t1.TradingDays.sum(), "Articles": t1.Articles.sum(),
       "PreOpen%": 100 * (news.timing == "pre-open").mean(), "Intraday%": 100 * (news.timing == "intraday").mean(),
       "PostClose%": 100 * (news.timing == "post-close").mean(), "NewsDays": t1.NewsDays.sum(),
       "NewsDaysTrain": t1.NewsDaysTrain.sum(), "NewsDaysTest": t1.NewsDaysTest.sum(),
       "MeanWordsPerArticle": news.News.str.split().str.len().mean()}
t1 = pd.concat([t1, pd.DataFrame([tot])], ignore_index=True); t1.to_csv(f"{OUT}/table1_dataset.csv", index=False)
R["n_articles_rows"] = int(len(news)); R["n_articles_unique"] = int(news.ArticleID.nunique())
R["date_min"] = str(news.PublishedOn.min().date()); R["date_max"] = str(news.PublishedOn.max().date())
R["price_min"] = str(prices.Date.min().date()); R["price_max"] = str(prices.Date.max().date())
R["n_price_rows"] = int(len(prices)); R["mean_words"] = float(tot["MeanWordsPerArticle"])
R["pre_open_pct"] = tot["PreOpen%"]; R["intraday_pct"] = tot["Intraday%"]; R["post_close_pct"] = tot["PostClose%"]
R["news_days_total"] = int(tot["NewsDays"]); R["news_days_train"] = int(tot["NewsDaysTrain"]); R["news_days_test"] = int(tot["NewsDaysTest"])
R["articles_by_year"] = news.groupby(news.PublishedOn.dt.year).size().to_dict()


TRc = pool("contemporaneous", is_train)
raw_tokens = TRc.News.str.split(); n_raw_tok = int(raw_tokens.str.len().sum()); raw_vocab = len(set(w for t in raw_tokens for w in t))
norm_tok = int(TRc.Tokens.str.len().sum()); norm_vocab = len(set(w for t in TRc.Tokens for w in t))
vec_tmp = TfidfVectorizer(analyzer=lambda x: x, min_df=MIN_DF).fit(TRc.Tokens); vocab_mindf = len(vec_tmp.vocabulary_)
lex_eq_pool = eqsplit_lexicon(TRc); vocab_eq = len(lex_eq_pool)
t2 = pd.DataFrame([["Raw whitespace tokens (training documents, 11 firms, 2018-2021)", n_raw_tok, raw_vocab],
                   ["After normalisation (diacritics, punctuation, alef/ya/ta-marbuta, digits -> NUM)", norm_tok, norm_vocab],
                   [f"Regression vocabulary (document frequency >= {MIN_DF})", "", vocab_mindf],
                   [f"Equal-split lexicon (>= {MIN_DAYS_EQ} distinct days)", "", vocab_eq]], columns=["Stage", "Tokens", "Vocabulary"])
t2.to_csv(f"{OUT}/table2_vocabulary.csv", index=False)
R.update(n_raw_tok=n_raw_tok, raw_vocab=raw_vocab, norm_tok=norm_tok, norm_vocab=norm_vocab, vocab_mindf=vocab_mindf, vocab_eq=vocab_eq)


TRp = pool("predictive", is_train)
M_c = ridge_fit(TRc); M_p = ridge_fit(TRp)
LEX_c = eqsplit_lexicon(TRc); LEX_p = eqsplit_lexicon(TRp)


vec, rd = M_c; names = np.array(vec.get_feature_names_out())
dfreq = np.asarray((vec.transform(TRc.Tokens) > 0).sum(0)).ravel()
tw = pd.DataFrame({"Word": names, "Coefficient": rd.coef_, "DocFreq": dfreq})
tw = tw[tw.DocFreq >= 50]     # avoid concatenation artefacts from embedded tables
neg = tw.sort_values("Coefficient").head(25); pos = tw.sort_values("Coefficient", ascending=False).head(25)
t3 = pd.DataFrame({"Positive word": pos.Word.values, "Coef (+)": pos.Coefficient.values, "DocFreq (+)": pos.DocFreq.values,
                   "Negative word": neg.Word.values, "Coef (-)": neg.Coefficient.values, "DocFreq (-)": neg.DocFreq.values})
t3.to_csv(f"{OUT}/table3_top_words.csv", index=False)
tw.sort_values("Coefficient", ascending=False).to_csv(f"{OUT}/lexicon_pooled_ridge_2018_2021.csv", index=False)
LEX_c.to_csv(f"{OUT}/lexicon_pooled_equalsplit_2018_2021.csv", index=False)


rows = []; SCORES = {}
for mode, M, LEX in [("contemporaneous", M_c, LEX_c), ("predictive", M_p, LEX_p)]:
    for est in ["Equal-split (firm)", "Equal-split (pooled)", "Ridge (firm)", "Ridge (pooled)"]:
        S_all, Y_all = [], []
        for c in CIDS:
            d = DOCS[mode][c]; tr, te = d[is_train(d)], d[is_test(d)]
            if est == "Equal-split (firm)": s = eqsplit_score(te, eqsplit_lexicon(tr))
            elif est == "Equal-split (pooled)": s = eqsplit_score(te, LEX)
            elif est == "Ridge (firm)": s = ridge_score(te, ridge_fit(tr))
            else: s = ridge_score(te, M)
            m = evaluate(s, te.cc); m.update(Alignment=mode, Estimator=est, CompanyID=c, Company=NAME[c]); rows.append(m)
            S_all.append(s); Y_all.append(te.cc.values)
            if est == "Ridge (pooled)": SCORES[(mode, c)] = (te, s)
        m = evaluate(np.concatenate(S_all), np.concatenate(Y_all)); m.update(Alignment=mode, Estimator=est, CompanyID="", Company="Pooled (all firms)"); rows.append(m)
t4 = pd.DataFrame(rows); t4.to_csv(f"{OUT}/table4_main_results.csv", index=False)
for mode in ["contemporaneous", "predictive"]:
    for est in ["Equal-split (pooled)", "Ridge (firm)", "Ridge (pooled)"]:
        q = t4[(t4.Alignment == mode) & (t4.Estimator == est) & (t4.Company == "Pooled (all firms)")].iloc[0]
        R[f"{mode}_{est}"] = {k: (float(q[k]) if isinstance(q[k], (float, np.floating)) else q[k]) for k in ["n", "r", "nw_t", "nw_p", "spearman", "acc", "bal_acc", "auc", "majority", "pt_p"]}
    q = t4[(t4.Alignment == mode) & (t4.Estimator == "Ridge (pooled)") & (t4.Company != "Pooled (all firms)")]
    R[f"{mode}_ridge_pooled_firm_r_min"] = float(q.r.min()); R[f"{mode}_ridge_pooled_firm_r_max"] = float(q.r.max())
    R[f"{mode}_ridge_pooled_firms_positive"] = int((q.r > 0).sum()); R[f"{mode}_ridge_pooled_firms_sig05"] = int(((q.r > 0) & (q.nw_p < 0.05)).sum())
# in-sample (training) same-day r for the equal-split estimator, per firm — shows the circularity effect
ins = []
for c in CIDS:
    d = DOCS["contemporaneous"][c]; tr = d[is_train(d)]; s = eqsplit_score(tr, eqsplit_lexicon(tr)); ins.append(stats.pearsonr(s, tr.cc)[0])
R["insample_eqsplit_r_min"] = float(min(ins)); R["insample_eqsplit_r_max"] = float(max(ins))


rows = []
for mode in ["contemporaneous", "predictive"]:
    for Y in [2019, 2020, 2021, 2022]:
        TR = pool(mode, lambda d: d.Date.dt.year < Y); M = ridge_fit(TR)
        S, Yv = [], []
        for c in CIDS:
            te = DOCS[mode][c]; te = te[te.Date.dt.year == Y]
            if len(te) < 10: continue
            S.append(ridge_score(te, M)); Yv.append(te.cc.values)
        m = evaluate(np.concatenate(S), np.concatenate(Yv)); m.update(Alignment=mode, TestYear=Y, TrainYears=f"2018-{Y-1}"); rows.append(m)
t5 = pd.DataFrame(rows); t5.to_csv(f"{OUT}/table5_walkforward.csv", index=False)
R["walkforward"] = t5[["Alignment", "TestYear", "n", "r", "nw_t", "nw_p", "bal_acc", "majority", "pt_p"]].to_dict("records")


rows = []
for mode, M in [("contemporaneous", M_c), ("predictive", M_p)]:
    TR = pool(mode, is_train); TE = {c: DOCS[mode][c][is_test(DOCS[mode][c])] for c in CIDS}
    Yv = np.concatenate([TE[c].cc.values for c in CIDS])
    real = evaluate(np.concatenate([ridge_score(TE[c], M) for c in CIDS]), Yv)["r"]
    null = []
    for b in range(50):
        T2 = TR.copy(); T2["cc"] = rng.permutation(TR.cc.values); Mb = ridge_fit(T2)
        null.append(evaluate(np.concatenate([ridge_score(TE[c], Mb) for c in CIDS]), Yv)["r"])
    null = np.array(null)
    rows.append({"Alignment": mode, "Observed r": real, "Placebo mean": null.mean(), "Placebo 2.5%": np.percentile(null, 2.5),
                 "Placebo 97.5%": np.percentile(null, 97.5), "Placebo max": null.max(), "P(placebo >= observed)": (null >= real).mean(), "Permutations": 50})
    np.save(f"{OUT}/placebo_null_{mode}.npy", null)
t6 = pd.DataFrame(rows); t6.to_csv(f"{OUT}/table6_placebo.csv", index=False); R["placebo"] = t6.to_dict("records")


rows = []
for cls in ["pre-open", "intraday", "post-close"]:
    S, Yv = [], []
    for c in CIDS:
        p = prices[prices.CompanyID == c].sort_values("Date"); n = news[(news.CompanyID == c) & (news.timing == cls)].copy()
        n["Date"] = n.PublishedOn.dt.normalize()
        d = n.groupby("Date").News.apply(" ".join).reset_index().merge(p[["Date", "cc"]], on="Date"); d = d[d.Date > TRAIN_END]
        if len(d) < 5: continue
        d["Tokens"] = d.News.apply(tok); S.append(ridge_score(d, M_c)); Yv.append(d.cc.values)
    m = evaluate(np.concatenate(S), np.concatenate(Yv)); m["Publication time"] = cls; rows.append(m)
t7 = pd.DataFrame(rows); t7.to_csv(f"{OUT}/table7_timing.csv", index=False); R["timing"] = t7.to_dict("records")


TE = pool("predictive", is_test); s = ridge_score(TE, M_p); rows = []
for k in range(0, H + 1):
    m = evaluate(s, TE[f"f{k}"], lags=1); rows.append({"Horizon": f"session +{k}", "r": m["r"], "nw_t": m["nw_t"], "nw_p": m["nw_p"], "n": m["n"]})
for h in [2, 3, 5, 10]:
    car = sum(TE[f"f{k}"] for k in range(1, h + 1)); m = evaluate(s, car, lags=h)
    rows.append({"Horizon": f"cumulative +1..+{h}", "r": m["r"], "nw_t": m["nw_t"], "nw_p": m["nw_p"], "n": m["n"]})
t8 = pd.DataFrame(rows); t8.to_csv(f"{OUT}/table8_horizon.csv", index=False); R["horizon"] = t8.to_dict("records")


TRb = pool("predictive", is_train); TEb = pool("predictive", is_test)
ctrl = ["cc_l1", "cc_l2", "cc_l3", "cc_l4", "cc_l5", "dlv_l1", "mkt_l1"]
TRb = TRb.dropna(subset=ctrl); TEb = TEb.dropna(subset=ctrl).reset_index(drop=True)
lex_te = ridge_score(TEb, M_p); lex_tr = ridge_score(TRb, M_p)
y = TEb.cc.values; ydir = (y > 0).astype(int)
rows = []
maj = TRb.cc.gt(0).mean() > 0.5; s_maj = np.full(len(y), 1.0 if maj else -1.0)
rows.append({"Model": "B0 Majority direction of training window", "acc": (np.sign(y) == s_maj).mean(), "bal_acc": 0.5, "r": np.nan, "auc": 0.5})
accs = [(np.sign(y) == rng.choice([1, -1], size=len(y), p=[TRb.cc.gt(0).mean(), 1 - TRb.cc.gt(0).mean()])).mean() for _ in range(1000)]
rows.append({"Model": "B1 Random with training class prior (1,000 draws, mean)", "acc": np.mean(accs), "bal_acc": 0.5, "r": np.nan, "auc": 0.5})
ar = LinearRegression().fit(TRb[["cc_l1", "cc_l2", "cc_l3", "cc_l4", "cc_l5"]], TRb.cc); s_ar = ar.predict(TEb[["cc_l1", "cc_l2", "cc_l3", "cc_l4", "cc_l5"]])
m = evaluate(s_ar, y); rows.append({"Model": "B2 AR(5) on lagged returns", **{k: m[k] for k in ["acc", "bal_acc", "r", "auc"]}, "nw_t": m["nw_t"]})
ols = LinearRegression().fit(TRb[ctrl], TRb.cc); s_ct = ols.predict(TEb[ctrl])
m = evaluate(s_ct, y); rows.append({"Model": "B3 Lagged returns + lagged volume change + lagged market proxy", **{k: m[k] for k in ["acc", "bal_acc", "r", "auc"]}, "nw_t": m["nw_t"]})
m = evaluate(lex_te, y); rows.append({"Model": "Reversed lexicon (pooled ridge), pre-session information only", **{k: m[k] for k in ["acc", "bal_acc", "r", "auc"]}, "nw_t": m["nw_t"], "pt_p": m["pt_p"]})
Xtr = np.column_stack([TRb[ctrl].values, lex_tr]); Xte = np.column_stack([TEb[ctrl].values, lex_te])
both = LinearRegression().fit(Xtr, TRb.cc); s_both = both.predict(Xte)
m = evaluate(s_both, y); rows.append({"Model": "B3 + reversed lexicon", **{k: m[k] for k in ["acc", "bal_acc", "r", "auc"]}, "nw_t": m["nw_t"]})
t9 = pd.DataFrame(rows); t9.to_csv(f"{OUT}/table9_benchmarks.csv", index=False); R["benchmarks"] = t9.to_dict("records")
# incremental regression with HAC errors on the test sample: cc = a + b'controls + g*lexicon
X = sm.add_constant(pd.DataFrame(Xte, columns=ctrl + ["lexicon"])); f = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
f0 = sm.OLS(y, sm.add_constant(TEb[ctrl])).fit()
R["incremental"] = {"coef_lexicon": float(f.params["lexicon"]), "t_lexicon": float(f.tvalues["lexicon"]), "p_lexicon": float(f.pvalues["lexicon"]),
                    "r2_controls": float(f0.rsquared), "r2_with_lexicon": float(f.rsquared), "n": int(len(y))}
# OOS R2 (Campbell-Thompson) vs training-mean benchmark
mean_tr = TRb.cc.mean()
R["oos_r2_lexicon"] = float(1 - np.sum((y - lex_te) ** 2) / np.sum((y - mean_tr) ** 2))
R["oos_r2_controls"] = float(1 - np.sum((y - s_ct) ** 2) / np.sum((y - mean_tr) ** 2))
# Granger-style: does lexicon score help explain next-session return beyond lags, and does past return explain the score?
rev = sm.OLS(lex_te, sm.add_constant(TEb[["cc_l1", "cc_l2", "cc_l3", "cc_l4", "cc_l5"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
R["reverse_direction"] = {"F_p": float(rev.f_pvalue), "r2": float(rev.rsquared)}


rows = []
def run_variant(label, mode="contemporaneous", ycol="cc", drop_move=False, drop_wrap=False, alpha=ALPHA, min_df=MIN_DF):
    global ALPHA, MIN_DF
    a0, d0 = ALPHA, MIN_DF; ALPHA, MIN_DF = alpha, min_df
    dd = {c: documents(c, mode, drop_move, drop_wrap) for c in CIDS}
    TR = pd.concat([dd[c][is_train(dd[c])] for c in CIDS]); M = ridge_fit(TR, ycol)
    S = np.concatenate([ridge_score(dd[c][is_test(dd[c])], M) for c in CIDS]); Yv = np.concatenate([dd[c][is_test(dd[c])][ycol].values for c in CIDS])
    m = evaluate(S, Yv); ALPHA, MIN_DF = a0, d0
    rows.append({"Variant": label, "n": m["n"], "r": m["r"], "nw_t": m["nw_t"], "bal_acc": m["bal_acc"], "majority": m["majority"]})
run_variant("Baseline: close-to-close return, all articles")
run_variant("Open-to-close return instead of close-to-close", ycol="oc")
run_variant("Ridge penalty alpha = 1", alpha=1.0)
run_variant("Ridge penalty alpha = 100", alpha=100.0)
run_variant("Vocabulary threshold min_df = 20", min_df=20)
run_variant("Market-wrap articles removed", drop_wrap=True)
run_variant("Explicit price-movement words removed", drop_move=True)
run_variant("Both removed", drop_move=True, drop_wrap=True)
t10 = pd.DataFrame(rows); t10.to_csv(f"{OUT}/table10_robustness.csv", index=False); R["robustness"] = t10.to_dict("records")


rows = []
for c in CIDS:
    d = DOCS["contemporaneous"][c]; tr, te = d[is_train(d)], d[is_test(d)]
    r_firm = evaluate(ridge_score(te, ridge_fit(tr)), te.cc)["r"]; r_pool = evaluate(ridge_score(te, M_c), te.cc)["r"]
    others = pool("contemporaneous", lambda x: is_train(x) & (x.CompanyID != c)); r_loo = evaluate(ridge_score(te, ridge_fit(others)), te.cc)["r"]
    rows.append({"CompanyID": c, "Company": NAME[c], "n_test": len(te), "Firm-specific lexicon r": r_firm, "Pooled lexicon r": r_pool, "Leave-one-firm-out lexicon r": r_loo})
t11 = pd.DataFrame(rows); t11.to_csv(f"{OUT}/table11_transfer.csv", index=False); R["transfer"] = t11.to_dict("records")
R["transfer_means"] = {k: float(t11[k].mean()) for k in ["Firm-specific lexicon r", "Pooled lexicon r", "Leave-one-firm-out lexicon r"]}


TEt = pool("predictive", is_test).sort_values(["Date", "CompanyID"]); s_t = ridge_score(TEt, M_p); TEt["signal"] = s_t
daily = TEt.groupby("Date").apply(lambda g: pd.Series({"strat": g.loc[g.signal > 0, "oc"].mean() if (g.signal > 0).any() else 0.0,
                                                         "bh": g.oc.mean(), "n_long": int((g.signal > 0).sum())})).reset_index()
cal = prices[prices.Date > TRAIN_END].groupby("Date").oc.mean().rename("bh_all").reset_index()
daily = cal.merge(daily, on="Date", how="left").fillna({"strat": 0.0, "n_long": 0})
def perf(r, cost_bp=0.0, trades=None):
    r = r.copy()
    if trades is not None: r = r - cost_bp / 1e4 * (trades > 0)
    ann = 250; mu = r.mean() * ann; sd = r.std() * np.sqrt(ann); cum = np.exp(r.cumsum()); dd = (cum / cum.cummax() - 1).min()
    return {"Annualised return": mu, "Annualised volatility": sd, "Sharpe": mu / sd if sd > 0 else np.nan, "Max drawdown": dd, "Days in market": int((trades > 0).sum()) if trades is not None else len(r)}
rows = [{"Strategy": "Buy-and-hold, equal-weight 11 firms (open-to-close)", **perf(daily.bh_all)},
        {"Strategy": "Long/cash on positive lexicon signal, no cost", **perf(daily.strat, 0, daily.n_long)},
        {"Strategy": "Long/cash, 10 bp round-trip cost", **perf(daily.strat, 10, daily.n_long)},
        {"Strategy": "Long/cash, 25 bp round-trip cost", **perf(daily.strat, 25, daily.n_long)}]
t12 = pd.DataFrame(rows); t12.to_csv(f"{OUT}/table12_trading.csv", index=False); R["trading"] = t12.to_dict("records")


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
# Fig A: timing alignment diagram
fig, ax = plt.subplots(figsize=(8, 2.6)); ax.set_xlim(0, 48); ax.set_ylim(0, 3); ax.axis("off")
for x0, lab in [(0, "Day t"), (24, "Day t+1")]:
    ax.add_patch(plt.Rectangle((x0 + 10, 1.2), 5, 0.6, color="#cfe2f3")); ax.text(x0 + 12.5, 1.5, "session\n10:00-15:00", ha="center", va="center", fontsize=7)
    ax.plot([x0, x0 + 24], [1.2, 1.2], color="k", lw=0.8); ax.text(x0 + 12, 0.85, lab, ha="center", fontsize=8)
for x, txt, tgt in [(4, "pre-open article\n(published 08:00)", 12.5), (13, "intraday article\n(12:00)", 36.5), (19, "post-close article\n(16:00)", 36.5)]:
    ax.annotate("", xy=(tgt, 1.85), xytext=(x, 2.7), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1)); ax.text(x, 2.75, txt, ha="center", fontsize=7)
ax.text(24, 0.3, "Predictive alignment: each article is assigned to the first session that opens after its publication time", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_timing_alignment.png", dpi=200); plt.close()
# Fig B: walk-forward scheme
fig, ax = plt.subplots(figsize=(7, 2.2)); yrs = [2018, 2019, 2020, 2021, 2022]
for i, Y in enumerate([2019, 2020, 2021, 2022]):
    for j, yr in enumerate(yrs):
        col = "#4472c4" if yr < Y else ("#e67e22" if yr == Y else "white")
        ax.add_patch(plt.Rectangle((j, 3 - i), 1, 0.8, facecolor=col, edgecolor="grey")); ax.text(j + 0.5, 3.4 - i, str(yr), ha="center", va="center", fontsize=8, color="white" if col != "white" else "grey")
    ax.text(-0.2, 3.4 - i, f"Fold {i+1}", ha="right", va="center", fontsize=8)
ax.set_xlim(-1.2, 5.2); ax.set_ylim(-0.2, 4); ax.axis("off")
ax.add_patch(plt.Rectangle((0, -0.15), 0.4, 0.3, color="#4472c4")); ax.text(0.5, 0, "train (lexicon estimated)", va="center", fontsize=7)
ax.add_patch(plt.Rectangle((2.6, -0.15), 0.4, 0.3, color="#e67e22")); ax.text(3.1, 0, "test (lexicon frozen)", va="center", fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_walkforward.png", dpi=200); plt.close()
# Fig C: lexicon coefficient distribution (pooled ridge) + equal-split for comparison
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
axes[0].hist(rd.coef_, bins=80, color="#4472c4"); axes[0].set_title(f"Ridge lexicon coefficients (n = {len(rd.coef_):,} words)", fontsize=8); axes[0].set_xlabel("coefficient"); axes[0].set_ylabel("words")
axes[1].hist(LEX_c.Score, bins=80, color="#7f7f7f"); axes[1].set_title(f"Equal-split lexicon scores (n = {len(LEX_c):,} words)", fontsize=8); axes[1].set_xlabel("score")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_lexicon_distribution.png", dpi=200); plt.close()
# Fig D: per-firm OOS r, contemporaneous and predictive, pooled ridge, 2022
q = t4[(t4.Estimator == "Ridge (pooled)") & (t4.Company != "Pooled (all firms)")]
fig, ax = plt.subplots(figsize=(8, 3)); x = np.arange(len(CIDS)); w = 0.38
qc = q[q.Alignment == "contemporaneous"].set_index("CompanyID").loc[CIDS]; qp = q[q.Alignment == "predictive"].set_index("CompanyID").loc[CIDS]
ax.bar(x - w / 2, qc.r, w, label="same-session association", color="#4472c4"); ax.bar(x + w / 2, qp.r, w, label="next-session (pre-session information)", color="#e67e22")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([NAME[c].replace(" Co.", "").replace("Saudi ", "S. ")[:18] for c in CIDS], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("out-of-sample Pearson r (2022)"); ax.legend(fontsize=7, frameon=False); plt.tight_layout(); plt.savefig(f"{OUT}/fig_per_firm_r.png", dpi=200); plt.close()
# Fig E: walk-forward r by year
fig, ax = plt.subplots(figsize=(5.5, 2.8))
for mode, col in [("contemporaneous", "#4472c4"), ("predictive", "#e67e22")]:
    q = t5[t5.Alignment == mode]; ax.plot(q.TestYear, q.r, marker="o", color=col, label="same-session association" if mode == "contemporaneous" else "next-session (pre-session information)")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks([2019, 2020, 2021, 2022]); ax.set_ylabel("out-of-sample r"); ax.set_xlabel("test year (lexicon trained on all prior years)"); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_walkforward_r.png", dpi=200); plt.close()
# Fig F: placebo distribution
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
for ax, mode in zip(axes, ["contemporaneous", "predictive"]):
    null = np.load(f"{OUT}/placebo_null_{mode}.npy"); real = R[f"{mode}_Ridge (pooled)"]["r"]
    ax.hist(null, bins=20, color="#bfbfbf", label="lexicons trained on shuffled returns"); ax.axvline(real, color="#c0392b", lw=1.5, label=f"observed r = {real:.3f}")
    ax.set_title("same-session association" if mode == "contemporaneous" else "next-session (pre-session information)", fontsize=8); ax.set_xlabel("out-of-sample r, 2022"); ax.legend(fontsize=6, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_placebo.png", dpi=200); plt.close()
# Fig G: horizon
fig, ax = plt.subplots(figsize=(6, 2.8)); q = t8[t8.Horizon.str.startswith("session")]
ax.bar(range(len(q)), q.r, color=["#e67e22"] + ["#7f7f7f"] * (len(q) - 1)); ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(q))); ax.set_xticklabels([h.replace("session ", "") for h in q.Horizon], fontsize=7); ax.set_xlabel("sessions after the news is published"); ax.set_ylabel("r with session return")
for i, (r_, t_) in enumerate(zip(q.r, q.nw_t)): ax.text(i, r_ + (0.004 if r_ >= 0 else -0.012), f"t={t_:.1f}", ha="center", fontsize=6)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_horizon.png", dpi=200); plt.close()
# Fig H: pooled scatter, contemporaneous 2022
TEc = pool("contemporaneous", is_test); sc = ridge_score(TEc, M_c)
fig, ax = plt.subplots(figsize=(4.5, 3.6)); ax.scatter(sc, TEc.cc, s=6, alpha=0.4, color="#4472c4")
b = np.polyfit(sc, TEc.cc, 1); xs = np.linspace(sc.min(), sc.max(), 50); ax.plot(xs, np.polyval(b, xs), color="#c0392b", lw=1)
ax.set_xlabel("lexicon score (frozen, trained 2018-2021)"); ax.set_ylabel("close-to-close log return, same session"); ax.set_title(f"2022, {len(sc):,} firm-days, r = {R['contemporaneous_Ridge (pooled)']['r']:.3f}", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_scatter.png", dpi=200); plt.close()
# Fig I: redrawn Figure 3 (return computation) and Figure 5 (word contribution) with the new definitions
fig, ax = plt.subplots(figsize=(6, 2.4)); ax.axis("off")
for x, txt, col in [(0.08, "Previous close\n$C_{t-1}$", "#cfe2f3"), (0.38, "Close\n$C_t$", "#cfe2f3")]:
    ax.add_patch(plt.Rectangle((x, 0.6), 0.2, 0.3, color=col)); ax.text(x + 0.1, 0.75, txt, ha="center", va="center", fontsize=8)
ax.add_patch(plt.Rectangle((0.68, 0.55), 0.28, 0.4, color="#fde9d9")); ax.text(0.82, 0.75, "$r_t=\\ln(C_t/C_{t-1})$\nlog return", ha="center", va="center", fontsize=8)
ax.annotate("", xy=(0.68, 0.75), xytext=(0.58, 0.75), arrowprops=dict(arrowstyle="->")); ax.annotate("", xy=(0.38, 0.75), xytext=(0.28, 0.75), arrowprops=dict(arrowstyle="->"))
ax.text(0.5, 0.25, "$r_t>0$: positive session; $r_t<0$: negative session. Open-to-close $\\ln(C_t/O_t)$ used as robustness.", ha="center", fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_return.png", dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 3)); ax.axis("off")
ax.add_patch(plt.Rectangle((0.05, 0.75), 0.4, 0.18, color="#e4dff5")); ax.text(0.25, 0.84, "Session return: $r_t=+0.030$", ha="center", va="center", fontsize=8)
ax.add_patch(plt.Rectangle((0.55, 0.75), 0.4, 0.18, color="#e4dff5")); ax.text(0.75, 0.84, "Words in the day's document: $M_t=5$", ha="center", va="center", fontsize=8)
ax.add_patch(plt.Rectangle((0.25, 0.42), 0.5, 0.2, color="#c9e4e6")); ax.text(0.5, 0.52, "Equal-split contribution: $s(w_{t,i})=r_t/M_t=+0.006$", ha="center", va="center", fontsize=8)
for i, w in enumerate(["ارتفعت", "مبيعات", "الشركه", "بشكل", "ملحوظ"]):
    ax.add_patch(plt.Rectangle((0.03 + i * 0.19, 0.05), 0.17, 0.2, color="#fbe5c8")); ax.text(0.115 + i * 0.19, 0.15, f"{w}\n+0.006", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.115 + i * 0.19, 0.26), xytext=(0.5, 0.42), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.annotate("", xy=(0.5, 0.62), xytext=(0.25, 0.75), arrowprops=dict(arrowstyle="->", lw=0.6)); ax.annotate("", xy=(0.5, 0.62), xytext=(0.75, 0.75), arrowprops=dict(arrowstyle="->", lw=0.6))
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_contribution.png", dpi=200); plt.close()

# ------------------------------------------------------------------ save key numbers
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
json.dump(clean(R), open(f"{OUT}/key_numbers.json", "w"), indent=1, ensure_ascii=False)
print("done; outputs in", OUT)

