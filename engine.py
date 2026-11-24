"""ZeroC-AI analytics engine: data access, NLP classification, Pareto, recurrence, relationships,
SPC early warning, complaint-risk prediction, preventive-action recommender and KPIs."""
import os, sqlite3, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics.pairwise import linear_kernel

DB = os.path.join(os.path.dirname(__file__), "data", "complaints.db")
CATS = ["Quality deviation", "Physical defect", "Packaging", "Foreign matter", "Infestation", "Sensory",
        "Weight/Quantity", "Process defect", "Other"]
ROOTS = ["Man", "Machine", "Material", "Method", "Measurement", "Environment"]
PREVENTIVE_CHECKS = {   # SOP-style checks used by the rule layer of the recommender
 "Foreign matter": ["Test metal detector with certified test pieces at start, mid and end of shift", "Inspect sieves / magnets for wear before every batch"],
 "Infestation": ["Check warehouse temperature/humidity log and FIFO compliance", "Verify packing moisture against specification before release"],
 "Packaging": ["Verify sealer temperature/pressure set-points and leak-test sample packs", "Inspect sealing jaws and film reel specification"],
 "Sensory": ["Sensory-approve incoming raw material lot before use", "Confirm process time-temperature against validated recipe"],
 "Weight/Quantity": ["Verify check-weigher calibration with test weights", "Review net-content SPC chart for drift"],
 "Quality deviation": ["Hold lot until in-process QC results are within limits", "Cross-check lab instrument calibration"],
 "Physical defect": ["Inspect conveying/handling points for damage", "Review shipper strength and stacking pattern"],
 "Process defect": ["Confirm batch-card parameters before start-up", "Audit label/date-code printing at line start"],
 "Other": ["Review order documentation against customer agreement"]}

# ---------------------------------------------------------------- data access
def connect(db=DB): return sqlite3.connect(db)

def load(db=DB):
    con = connect(db)
    df = pd.read_sql("select * from complaints", con, parse_dates=["date"])
    vol = pd.read_sql("select * from volumes", con)
    con.close()
    df["month"] = df["date"].dt.to_period("M")
    vol["month"] = pd.PeriodIndex(vol["month"], freq="M")
    return df, vol

def add_complaint(rec, db=DB):
    con = connect(db); cur = con.cursor()
    n = cur.execute("select count(*) from complaints").fetchone()[0]
    rec = dict(rec); rec["complaint_id"] = f"C{n+1:05d}"
    cols = [r[1] for r in cur.execute("pragma table_info(complaints)")]
    vals = [rec.get(c) for c in cols]
    cur.execute(f"insert into complaints ({','.join(cols)}) values ({','.join('?'*len(cols))})", vals)
    con.commit(); con.close(); return rec["complaint_id"]

# ---------------------------------------------------------------- NLP classification
def _clean(s): return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())

class TextClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(preprocessor=_clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=2000, C=6, class_weight="balanced")
    def fit(self, df):
        X = self.vec.fit_transform(df["complaint_text"]); self.clf.fit(X, df["category"]); return self
    def predict_proba(self, text):
        p = self.clf.predict_proba(self.vec.transform([text]))[0]
        return pd.Series(p, index=self.clf.classes_).sort_values(ascending=False)
    def top_terms(self, k=6):
        names = np.array(self.vec.get_feature_names_out()); out = {}
        for i, c in enumerate(self.clf.classes_): out[c] = list(names[np.argsort(self.clf.coef_[i])[::-1][:k]])
        return out

def nlp_report(df):
    """Time-based hold-out evaluation (train on oldest 80%, test on newest 20%)."""
    d = df.sort_values("date"); cut = int(len(d) * .8); tr, te = d.iloc[:cut], d.iloc[cut:]
    m = TextClassifier().fit(tr); pred = m.clf.predict(m.vec.transform(te["complaint_text"]))
    rep = dict(accuracy=accuracy_score(te["category"], pred), macro_f1=f1_score(te["category"], pred, average="macro"),
               n_train=len(tr), n_test=len(te))
    full = TextClassifier().fit(d)
    return full, rep

def theme_clusters(df, k=10, seed=0):
    vec = TfidfVectorizer(preprocessor=_clean, stop_words="english", min_df=3, ngram_range=(1, 2))
    X = vec.fit_transform(df["complaint_text"]); km = KMeans(k, n_init=5, random_state=seed).fit(X)
    names = np.array(vec.get_feature_names_out()); rows = []
    for i in range(k):
        idx = np.where(km.labels_ == i)[0]; cen = km.cluster_centers_[i]
        top = ", ".join(names[np.argsort(cen)[::-1][:5]])
        dom = df.iloc[idx]["category"].value_counts()
        rows.append(dict(cluster=i + 1, size=len(idx), top_terms=top, dominant_category=dom.index[0], purity=round(dom.iloc[0] / len(idx), 2)))
    return pd.DataFrame(rows).sort_values("size", ascending=False)

# ---------------------------------------------------------------- Pareto / recurrence / relationships
def pareto(df, by, top=None):
    s = df[by].value_counts(); t = s.to_frame("complaints"); t["share_%"] = (t["complaints"] / t["complaints"].sum() * 100).round(1)
    t["cum_%"] = t["share_%"].cumsum().round(1)
    t["vital_few"] = t["cum_%"].shift(fill_value=0) < 80
    return t.head(top) if top else t

def recurrence(df, asof=None, window_m=24):
    asof = asof or df["date"].max(); lo = asof - pd.DateOffset(months=window_m); lo2 = asof - pd.DateOffset(months=2 * window_m)
    rows = []
    for (p, c), g in df.groupby(["product", "category"]):
        g = g.sort_values("date"); rec = g[g["date"] > lo]; prev = g[(g["date"] <= lo) & (g["date"] > lo2)]
        gaps = g["date"].diff().dt.days.dropna(); known = g.dropna(subset=["action_effective"])
        rows.append(dict(product=p, category=c, total=len(g), last_24m=len(rec), prev_24m=len(prev),
                         trend_pct=round(100 * (len(rec) - len(prev)) / max(len(prev), 1), 0),
                         median_gap_days=float(gaps.median()) if len(gaps) else np.nan, last_seen=g["date"].max().date(),
                         actions_effective_pct=round(100 * (known["action_effective"] == "Y").mean(), 0) if len(known) else np.nan))
    r = pd.DataFrame(rows)
    r["status"] = np.where(r["last_24m"] >= 10, "Chronic", np.where(r["last_24m"] >= 5, "Recurring", "Occasional"))
    return r.sort_values(["last_24m", "total"], ascending=False).reset_index(drop=True)

def lift_matrix(df, row, col="category"):
    ct = pd.crosstab(df[row], df[col]); p_c = ct.sum(0) / ct.values.sum(); p_r = ct.sum(1) / ct.values.sum()
    exp = np.outer(p_r, p_c) * ct.values.sum(); return (ct / exp).round(2), ct

def rules(df, min_count=20):
    """Association rules  antecedent -> category  with support, confidence, lift."""
    n = len(df); rows = []
    base = df["category"].value_counts(normalize=True)
    for col in ["product", "customer", "line", "root_cause", "customer_type"]:
        d = df.dropna(subset=[col])
        for (a, c), g in d.groupby([col, "category"]):
            na = (d[col] == a).sum()
            if len(g) >= min_count:
                conf = len(g) / na; rows.append(dict(antecedent=f"{col} = {a}", consequent=c, count=len(g), support=round(len(g) / n, 4),
                                                       confidence=round(conf, 3), lift=round(conf / base[c], 2)))
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)

def root_cause_table(df):
    d = df.dropna(subset=["root_cause"]); return pd.crosstab(d["category"], d["root_cause"], normalize="index").round(3).reindex(columns=ROOTS, fill_value=0)

# ---------------------------------------------------------------- SPC early warning (u-chart on complaints / 100k units)
def spc(df, vol, product=None, baseline_m=36, recent_m=3):
    d = df if product is None else df[df["product"] == product]
    v = vol if product is None else vol[vol["product"] == product]
    m = d.groupby("month").size().rename("complaints").to_frame().join(v.groupby("month")["units_dispatched"].sum(), how="outer").fillna({"complaints": 0})
    m = m.dropna(); m["rate"] = m["complaints"] / m["units_dispatched"] * 100_000
    base = m.iloc[-(baseline_m + recent_m):-recent_m]
    ubar = base["complaints"].sum() / base["units_dispatched"].sum() * 100_000
    m["ucl"] = ubar + 3 * np.sqrt(ubar / (m["units_dispatched"] / 100_000)); m["cl"] = ubar
    m["signal"] = m["rate"] > m["ucl"]
    # CUSUM (k=0.5 sigma, h=4 sigma) on standardised rate
    sd = np.sqrt(ubar / (m["units_dispatched"] / 100_000)); z = (m["rate"] - ubar) / sd; cs, c = [], 0.0
    for zi in z: c = max(0, c + zi - .5); cs.append(c)
    m["cusum"] = cs; m["cusum_signal"] = m["cusum"] > 4
    return m

def spc_alerts(df, vol, recent_m=3):
    out = []
    for p in sorted(df["product"].unique()):
        m = spc(df, vol, p); r = m.iloc[-recent_m:]
        if r["signal"].any() or (r["cusum_signal"].iloc[-1] and m["rate"].iloc[-1] > m["cl"].iloc[-1]):
            out.append(dict(product=p, latest_rate=round(m["rate"].iloc[-1], 2), centre=round(m["cl"].iloc[-1], 2), ucl=round(m["ucl"].iloc[-1], 2),
                            rule="Rate above UCL" if r["signal"].any() else "CUSUM drift"))
    return pd.DataFrame(out)

# ---------------------------------------------------------------- risk model
def _panel(df, vol):
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M"); T = len(months)
    prods = sorted(df["product"].unique()); P = len(prods); C = len(CATS)
    pi = {p: i for i, p in enumerate(prods)}; ci = {c: i for i, c in enumerate(CATS)}; mi = {m: i for i, m in enumerate(months)}
    cnt = np.zeros((P, C, T), dtype=np.float32)
    g = df.groupby(["product", "category", "month"]).size()
    for (p, c, m), n in g.items(): cnt[pi[p], ci[c], mi[m]] = n
    vv = np.ones((P, T), dtype=np.float32)
    for r in vol.itertuples():
        if r.month in mi and r.product in pi: vv[pi[r.product], mi[r.month]] = r.units_dispatched
    for p in range(P):                       # months without volume data (e.g. newly logged month) inherit the last known volume
        for t in range(1, T):
            if vv[p, t] == 1: vv[p, t] = vv[p, t - 1]
    return months, prods, cnt, vv

def _features(cnt, vv, months):
    P, C, T = cnt.shape; cs = np.concatenate([np.zeros((P, C, 1), np.float32), cnt.cumsum(2)], 2)
    def roll(k):  # sum of months t-k+1..t  at each t
        out = np.zeros_like(cnt)
        for t in range(T): out[:, :, t] = cs[:, :, t + 1] - cs[:, :, max(0, t + 1 - k)]
        return out
    r1, r3, r6, r12, r36 = cnt, roll(3), roll(6), roll(12), roll(36)
    ptot3 = r3.sum(1, keepdims=True).repeat(C, 1); ctot3 = r3.sum(0, keepdims=True).repeat(P, 0)
    last = np.zeros_like(cnt)
    for p in range(P):
        for c in range(C):
            ls = -999
            for t in range(T):
                if cnt[p, c, t] > 0: ls = t
                last[p, c, t] = min(t - ls, 60)
    same = np.zeros_like(cnt); same[:, :, 11:] = cnt[:, :, :T - 11]        # month t+1-12 = t-11 -> same calendar month as target
    nxt = (months.month.values % 12) + 1                                   # target calendar month
    ang = 2 * np.pi * nxt / 12; msin = np.broadcast_to(np.sin(ang), (P, C, T)); mcos = np.broadcast_to(np.cos(ang), (P, C, T))
    lvol = np.log(np.broadcast_to(vv[:, None, :], (P, C, T)))
    pid = np.broadcast_to(np.arange(P)[:, None, None], (P, C, T)); cid = np.broadcast_to(np.arange(C)[None, :, None], (P, C, T))
    feats = {"pair_last1m": r1, "pair_last3m": r3, "pair_last6m": r6, "pair_last12m": r12, "pair_last36m": r36,
             "months_since_last": last, "same_month_last_year": same, "product_last3m_all": ptot3, "category_last3m_all": ctot3,
             "season_sin": msin, "season_cos": mcos, "log_volume": lvol, "product_id": pid, "category_id": cid}
    return feats

FEATURE_LABELS = {"pair_last1m": "complaints last month", "pair_last3m": "complaints last 3 months", "pair_last6m": "complaints last 6 months",
                  "pair_last12m": "complaints last 12 months", "pair_last36m": "complaints last 36 months", "months_since_last": "months since last complaint",
                  "same_month_last_year": "complaints in same month last year", "product_last3m_all": "product's total recent complaints",
                  "category_last3m_all": "category-wide recent complaints", "season_sin": "season", "season_cos": "season", "log_volume": "production volume",
                  "product_id": "product identity", "category_id": "category identity"}

class RiskModel:
    def __init__(self, df, vol, test_from="2024-01"):
        self.df, self.vol = df, vol
        self.months, self.prods, self.cnt, self.vv = _panel(df, vol)
        self.F = _features(self.cnt, self.vv, self.months); self.names = list(self.F)
        P, C, T = self.cnt.shape; self.P, self.C, self.T = P, C, T
        self.y = (np.concatenate([self.cnt[:, :, 1:], np.zeros((P, C, 1), np.float32)], 2) > 0).astype(int)   # complaint next month?
        tt = np.arange(T); self.valid = (tt >= 12) & (tt <= T - 2)
        self.test_start = list(self.months).index(pd.Period(test_from, "M"))
        self.report = self._fit_eval()

    def _X(self, t_idx):
        X = np.stack([self.F[n][:, :, t_idx].reshape(-1) for n in self.names], 1) if np.ndim(t_idx) == 0 else None
        return X

    def _matrix(self, ts):
        Xs, ys, meta = [], [], []
        for t in ts:
            Xs.append(np.stack([self.F[n][:, :, t].reshape(-1) for n in self.names], 1)); ys.append(self.y[:, :, t].reshape(-1))
            meta += [(t, p, c) for p in range(self.P) for c in range(self.C)]
        return np.vstack(Xs), np.concatenate(ys), meta

    def _new(self): return HistGradientBoostingClassifier(max_iter=250, learning_rate=.05, max_depth=4, l2_regularization=1.0,
                                                          categorical_features=[self.names.index("product_id"), self.names.index("category_id")], random_state=0)
    def _fit_eval(self):
        tr_t = [t for t in range(12, self.test_start - 1)]; te_t = [t for t in range(self.test_start, self.T - 1)]
        Xtr, ytr, _ = self._matrix(tr_t); Xte, yte, meta = self._matrix(te_t)
        m = self._new().fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
        base = Xte[:, self.names.index("pair_last6m")] + .01 * Xte[:, self.names.index("pair_last36m")]
        k = int(len(yte) * .10); top = np.argsort(-p)[:k]; topb = np.argsort(-base)[:k]
        # monthly precision@10 : alert top-10 product-category pairs each month
        meta = np.array(meta); hit_m, hit_b, rec = [], [], []
        for t in te_t:
            ix = np.where(meta[:, 0] == t)[0]; o = ix[np.argsort(-p[ix])[:10]]; ob = ix[np.argsort(-base[ix])[:10]]
            hit_m.append(yte[o].mean()); hit_b.append(yte[ob].mean()); rec.append(yte[o].sum() / max(1, yte[ix].sum()))
        self.test_pred = (meta, p, yte)
        rep = dict(train_months=len(tr_t), test_months=len(te_t), n_test=len(yte), base_rate=round(float(yte.mean()), 3),
                   roc_auc=round(roc_auc_score(yte, p), 3), pr_auc=round(average_precision_score(yte, p), 3),
                   pr_auc_baseline=round(average_precision_score(yte, base), 3),
                   brier=round(brier_score_loss(yte, p), 3), recall_top10pct=round(float(yte[top].sum() / yte.sum()), 3),
                   lift_top10pct=round(float(yte[top].mean() / yte.mean()), 2), precision_at10_model=round(float(np.mean(hit_m)), 3),
                   precision_at10_baseline=round(float(np.mean(hit_b)), 3), recall_at10=round(float(np.mean(rec)), 3))
        # final model on everything labelled
        Xa, ya, _ = self._matrix(range(12, self.T - 1)); self.model = self._new().fit(Xa, ya); self.X_all = Xa
        return rep

    def score_next(self):
        t = self.T - 1; X = np.stack([self.F[n][:, :, t].reshape(-1) for n in self.names], 1)
        p = self.model.predict_proba(X)[:, 1]; rows = []
        for i, (pp, cc) in enumerate([(p_, c_) for p_ in range(self.P) for c_ in range(self.C)]):
            rows.append(dict(product=self.prods[pp], category=CATS[cc], risk=float(p[i]), _x=X[i]))
        d = pd.DataFrame(rows); d["tier"] = pd.cut(d["risk"], [-1, .25, .5, 1.01], labels=["Low", "Medium", "High"])
        drv = []
        med = np.median(self.X_all, 0)
        for _, r in d.iterrows():
            x = r["_x"]; base_p = r["risk"]; contrib = []
            for j, n in enumerate(self.names):
                if n in ("product_id", "category_id", "season_cos"): continue
                x2 = x.copy(); x2[j] = med[j] if n != "season_sin" else med[j]
                contrib.append((base_p - self.model.predict_proba(x2[None])[0, 1], n, x[j]))
            contrib.sort(reverse=True); drv.append("; ".join(f"{FEATURE_LABELS[n]} = {v:.0f}" if n not in ("log_volume", "season_sin") else FEATURE_LABELS[n]
                                                            for c, n, v in contrib[:3] if c > 0.005))
        d["why"] = drv; return d.drop(columns="_x").sort_values("risk", ascending=False).reset_index(drop=True)

    def history(self, product, category):
        return pd.Series(self.cnt[self.prods.index(product), CATS.index(category)], index=self.months.astype(str))

    def importance(self, n=1500, seed=0):
        from sklearn.inspection import permutation_importance
        rng = np.random.default_rng(seed); te_t = [t for t in range(self.test_start, self.T - 1)]
        X, y, _ = self._matrix(te_t); ix = rng.choice(len(y), min(n, len(y)), replace=False)
        r = permutation_importance(self.model, X[ix], y[ix], scoring="average_precision", n_repeats=4, random_state=0)
        imp = pd.Series(r.importances_mean, index=[FEATURE_LABELS[k] for k in self.names]).groupby(level=0).sum().sort_values(ascending=False)
        return imp

# ---------------------------------------------------------------- recommender
class Recommender:
    def __init__(self, df):
        self.df = df.reset_index(drop=True); self.vec = TfidfVectorizer(preprocessor=_clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.X = self.vec.fit_transform(self.df["complaint_text"])
        self.known = self.df["corrective_action"].notna() & self.df["action_effective"].notna()
    def recommend(self, product, category, text="", top=5):
        d = self.df; sim = np.zeros(len(d))
        if text.strip(): sim = linear_kernel(self.vec.transform([text]), self.X)[0]
        w = (d["category"] == category).astype(float) * 1.0 + (d["product"] == product).astype(float) * .6 + sim * 1.2
        # recent history counts more (half-life 5 years)
        age = (d["date"].max() - d["date"]).dt.days / 365; w = w * np.power(.5, age / 5)
        sel = self.known & (d["category"] == category)
        g = d[sel].assign(w=w[sel]).groupby("corrective_action")
        rows = []
        for a, x in g:
            ws = x["w"].sum(); eff = (x["w"] * (x["action_effective"] == "Y")).sum()
            score = (eff + 1) / (ws + 2)                                   # Bayesian-smoothed weighted effectiveness
            rows.append(dict(action=a, evidence_cases=len(x), weighted_effectiveness=round(float(score), 2),
                             raw_effective_pct=round(100 * (x["action_effective"] == "Y").mean(), 0), score=score * np.log1p(len(x))))
        t = pd.DataFrame(rows).sort_values("score", ascending=False).head(top).drop(columns="score").reset_index(drop=True)
        # top similar past cases
        sc = np.argsort(-(w.values))[:5]
        cases = d.iloc[sc][["complaint_id", "date", "product", "category", "complaint_text", "root_cause", "corrective_action", "action_effective"]]
        return t, cases, PREVENTIVE_CHECKS.get(category, [])

# ---------------------------------------------------------------- KPIs
def kpis(df, vol, rm=None):
    last = df["month"].max(); l12 = df[df["month"] > last - 12]; p12 = df[(df["month"] <= last - 12) & (df["month"] > last - 24)]
    v12 = vol[(vol["month"] > last - 12)]["units_dispatched"].sum(); vp = vol[(vol["month"] <= last - 12) & (vol["month"] > last - 24)]["units_dispatched"].sum()
    ppm = lambda d, v: len(d) / v * 1e6
    rec = recurrence(df); 
    # repeat-complaint rate: share of last-12m complaints whose product+category had a complaint in the preceding 12 months
    rr = []
    for _, r in l12.iterrows():
        prev = df[(df["customer"] == r["customer"]) & (df["product"] == r["product"]) & (df["category"] == r["category"]) & (df["date"] < r["date"]) & (df["date"] >= r["date"] - pd.Timedelta(days=365))]
        rr.append(len(prev) > 0)
    ok = df[df["action_effective"].notna()]; ok12 = ok[ok["month"] > last - 24]
    streak = {}
    for (p, c), g in df.groupby(["product", "category"]):
        streak[f"{p} · {c}"] = int(last.ordinal - g["month"].max().ordinal)
    return dict(complaints_12m=len(l12), complaints_prev12m=len(p12), ppm_12m=round(ppm(l12, v12), 1), ppm_prev12m=round(ppm(p12, vp), 1),
                repeat_rate=round(100 * np.mean(rr), 1), capa_effective=round(100 * (ok12["action_effective"] == "Y").mean(), 1),
                avg_closure_days=round(l12["closure_days"].mean(), 1), prev_closure_days=round(p12["closure_days"].mean(), 1),
                streaks=pd.Series(streak).sort_values(ascending=False),
                chronic_pairs=int((rec["status"] == "Chronic").sum()), pairs_free_12m=int((pd.Series(streak) >= 12).sum()), n_pairs=len(streak))
