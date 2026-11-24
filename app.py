"""ZeroC-AI – Zero Customer Complaint Analytics & Prediction System (prototype dashboard).
Run:  streamlit run app.py"""
import os, datetime as dt
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
import engine as E

OR, DK, GR, LT, AM, GN = "#FF3300", "#222B38", "#6B7280", "#F4F5F7", "#F59E0B", "#2E9E6B"
st.set_page_config(page_title="ZeroC-AI | Zero Complaint Command Centre", page_icon="🎯", layout="wide")
st.markdown(f"""<style>
.block-container{{padding-top:1.4rem;max-width:1400px}}
h1,h2,h3{{color:{DK}}}
.zc-head{{background:{DK};color:white;padding:14px 22px;border-radius:8px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center}}
.zc-head b{{font-size:1.25rem}} .zc-tag{{color:#FFB199;font-size:.8rem;font-weight:700}}
.zc-alert{{border:1px solid #D9DCE1;border-radius:8px;padding:10px 14px;margin-bottom:8px;background:{LT}}}
.zc-pill{{display:inline-block;color:white;font-weight:700;font-size:.72rem;padding:2px 10px;border-radius:4px;margin-right:8px}}
.zc-sub{{color:{GR};font-size:.85rem}}
div[data-testid="stMetric"]{{background:{LT};border:1px solid #D9DCE1;border-radius:8px;padding:10px 14px}}
</style>""", unsafe_allow_html=True)
HEAT = [[0, "#EEF7F1"], [.5, "#FFE08A"], [1, OR]]
def fig_style(f, h=360):
    f.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=12, color=DK), legend=dict(orientation="h", y=-.2)); return f

# ------------------------------------------------------------------ cached model bundle
@st.cache_resource(show_spinner="Training models on complaint history …")
def bundle(version):
    df, vol = E.load()
    clf, nlp_rep = E.nlp_report(df)
    rm = E.RiskModel(df, vol); scores = rm.score_next()
    rec = E.recurrence(df); rules = E.rules(df)
    return dict(df=df, vol=vol, clf=clf, nlp=nlp_rep, rm=rm, scores=scores, rec=rec, rules=rules,
                recommender=E.Recommender(df), kpi=E.kpis(df, vol), spc=E.spc_alerts(df, vol), themes=E.theme_clusters(df), importance=rm.importance())
B = bundle(os.path.getmtime(E.DB)); df, vol = B["df"], B["vol"]; rm = B["rm"]
last_month = df["month"].max(); next_month = (last_month + 1).strftime("%b %Y")

# ------------------------------------------------------------------ header + nav
st.markdown(f"""<div class="zc-head"><div><b>🎯 ZeroC-AI</b> &nbsp;|&nbsp; Zero Customer Complaint Command Centre</div>
<div class="zc-tag">PS-036 PROTOTYPE · DATA: {"REAL" if os.environ.get("ZC_DATA")=="real" else "SYNTHETIC DEMO"}</div></div>""", unsafe_allow_html=True)
PAGES = ["Command Centre", "Pareto & Classification", "Chronic & Recurring", "Relationships & Root Cause", "Risk & Prediction",
         "Preventive Recommendations", "Log New Complaint", "KPIs: Road to Zero", "About & Method"]
with st.sidebar:
    st.markdown("### Navigation"); page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.markdown("---"); st.caption(f"**{len(df):,}** complaints · {df['date'].min():%b %Y} – {df['date'].max():%b %Y}")
    st.caption(f"{df['product'].nunique()} products · {df['customer'].nunique()} customers")
    st.caption(f"Forecast month: **{next_month}**")
    if st.button("🔄 Retrain models now"): st.cache_resource.clear(); st.rerun()

# ------------------------------------------------------------------ helpers
def top_action(product, category):
    t, _, _ = B["recommender"].recommend(product, category); return (t.iloc[0]["action"], t.iloc[0]["weighted_effectiveness"], t.iloc[0]["evidence_cases"]) if len(t) else (None, None, None)
def pill(txt, col): return f'<span class="zc-pill" style="background:{col}">{txt}</span>'
def build_alerts(thr):
    out = []; s = B["scores"]; rec = B["rec"].set_index(["product", "category"])
    for _, r in s[s["risk"] >= thr].head(12).iterrows():
        chronic = rec.loc[(r["product"], r["category"]), "status"] == "Chronic"
        lvl = "HIGH" if (r["risk"] >= .5 or (chronic and r["risk"] >= .45)) else "MEDIUM"
        a, e, n = top_action(r["product"], r["category"])
        out.append(dict(level=lvl, title=f"{r['product']} · {r['category']} · {r['risk']:.0%} chance of a complaint in {next_month}" + (" · chronic problem" if chronic else ""),
                        why=r["why"], action=a, evidence=f"{e:.0%} effective over {n} similar cases" if a else ""))
    for _, r in B["spc"].iterrows():
        out.append(dict(level="WATCH", title=f"{r['product']} · complaint rate drifting ({r['rule']}): {r['latest_rate']} vs centre line {r['centre']} per 100k units",
                        why="SPC early-warning rule triggered on monthly complaint rate", action="Review last 3 months of batches & process records for this product", evidence=""))
    return out

# ================================================================== PAGES
if page == "Command Centre":
    k = B["kpi"]; c = st.columns(5)
    c[0].metric("Complaints (last 12 m)", k["complaints_12m"], f"{k['complaints_12m']-k['complaints_prev12m']:+d} vs prior 12 m", delta_color="inverse")
    c[1].metric("Complaints / million units", k["ppm_12m"], f"{k['ppm_12m']-k['ppm_prev12m']:+.1f}", delta_color="inverse")
    c[2].metric("Repeat-complaint rate", f"{k['repeat_rate']}%"); c[3].metric("Chronic problem areas", k["chronic_pairs"])
    c[4].metric("Avg closure (days)", k["avg_closure_days"], f"{k['avg_closure_days']-k['prev_closure_days']:+.1f}", delta_color="inverse")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("🚨 Early-warning alerts & recommended actions")
        thr = st.slider("Risk threshold for alerts", .25, .70, .40, .05, help="Probability of at least one complaint in the next month for a product × category")
        col = {"HIGH": OR, "MEDIUM": AM, "WATCH": GN}
        for a in build_alerts(thr):
            st.markdown(f"""<div class="zc-alert">{pill(a['level'], col[a['level']])}<b>{a['title']}</b><br>
<span class="zc-sub">Why: {a['why']}</span><br>👉 <b>Action:</b> {a['action']} <span class="zc-sub">{a['evidence']}</span></div>""", unsafe_allow_html=True)
    with right:
        m = df.groupby("month").size().rename("complaints").to_frame(); m.index = m.index.to_timestamp()
        m["12-month average"] = m["complaints"].rolling(12).mean()
        f = go.Figure(); f.add_bar(x=m.index, y=m["complaints"], name="Monthly complaints", marker_color="#FFB199")
        f.add_scatter(x=m.index, y=m["12-month average"], name="12-month average", line=dict(color=DK, width=2)); f.update_layout(title="Complaint trend (19 years)")
        st.plotly_chart(fig_style(f, 300), width="stretch")
        s = B["scores"].pivot(index="product", columns="category", values="risk")[E.CATS]
        f = px.imshow(s, color_continuous_scale=HEAT, zmin=0, zmax=.7, aspect="auto", title=f"Risk heat-map · {next_month}"); f.update_traces(text=s.round(2).values, texttemplate="%{text}", textfont_size=9)
        st.plotly_chart(fig_style(f, 380), width="stretch")

elif page == "Pareto & Classification":
    st.subheader("1 · Complaint classification & 2 · Pareto – where do complaints come from?")
    dim = st.selectbox("Pareto by", ["category", "product", "customer", "line", "root_cause", "customer_type"], format_func=lambda x: x.replace("_", " ").title())
    yrs = st.slider("Years included", int(df["date"].dt.year.min()), int(df["date"].dt.year.max()), (2007, 2026)); d = df[df["date"].dt.year.between(*yrs)]
    p = E.pareto(d, dim); f = go.Figure()
    f.add_bar(x=p.index.astype(str), y=p["complaints"], marker_color=[OR if v else "#D9DCE1" for v in p["vital_few"]], name="Complaints")
    f.add_scatter(x=p.index.astype(str), y=p["cum_%"], yaxis="y2", name="Cumulative %", line=dict(color=DK)); f.add_hline(y=80, line_dash="dot", line_color=GR, yref="y2")
    f.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], tickmode="array", tickvals=[0, 20, 40, 60, 80, 100], showgrid=False), title=f"Pareto by {dim} – orange bars are the 'vital few' (first ~80%)")
    c1, c2 = st.columns([1.5, 1]); c1.plotly_chart(fig_style(f, 380), width="stretch"); c2.dataframe(p, width="stretch", height=380)
    t = d.groupby([d["date"].dt.year, "category"]).size().reset_index(name="n"); t.columns = ["year", "category", "n"]
    st.plotly_chart(fig_style(px.bar(t, x="year", y="n", color="category", title="Complaints per year by category", color_discrete_sequence=px.colors.qualitative.Set2), 330), width="stretch")
    st.markdown("#### NLP classifier – reads the complaint text and assigns the category")
    n = B["nlp"]; a, b, c_ = st.columns(3)
    a.metric("Accuracy (time-based hold-out)", f"{n['accuracy']:.1%}"); b.metric("Macro F1", f"{n['macro_f1']:.3f}"); c_.metric("Trained on / tested on", f"{n['n_train']:,} / {n['n_test']:,}")
    st.caption("Model: TF-IDF (1–2 word groups) + Logistic Regression. Trained on the oldest 80 % of complaints, tested on the newest 20 %. Scores on synthetic text are optimistic – re-measure on the real database.")
    tt = pd.DataFrame({k: ", ".join(v) for k, v in B["clf"].top_terms().items()}, index=["Strongest words"]).T; st.dataframe(tt, width="stretch")
    st.markdown("#### Theme discovery (unsupervised clustering)"); st.dataframe(B["themes"], width="stretch", hide_index=True)

elif page == "Chronic & Recurring":
    st.subheader("3 · Recurring complaints and chronic problem areas")
    r = B["rec"]; stt = st.multiselect("Status", ["Chronic", "Recurring", "Occasional"], default=["Chronic", "Recurring"])
    st.caption("Chronic = 10 or more complaints for the same product + category in the last 24 months; Recurring = 5–9.")
    show = r[r["status"].isin(stt)].rename(columns={"last_24m": "complaints_last_24m", "prev_24m": "complaints_prev_24m", "trend_pct": "trend_%", "actions_effective_pct": "actions_effective_%"})
    st.dataframe(show, width="stretch", hide_index=True, height=330)
    c1, c2 = st.columns(2); prods = sorted(df["product"].unique())
    p = c1.selectbox("Product", prods, index=1); cat = c2.selectbox("Category", E.CATS, index=3)
    h = rm.history(p, cat); h.index = pd.PeriodIndex(h.index, freq="M").to_timestamp()
    g = df[(df["product"] == p) & (df["category"] == cat)]
    f = go.Figure(); f.add_bar(x=h.index, y=h.values, marker_color=OR, name="complaints")
    mk = g[g["action_effective"] == "Y"]; f.add_scatter(x=mk["date"], y=[0] * len(mk), mode="markers", marker=dict(color=GN, size=6, symbol="triangle-up"), name="action recorded as effective")
    f.update_layout(title=f"{p} · {cat}: complaints per month and actions marked 'effective'")
    st.plotly_chart(fig_style(f, 320), width="stretch")
    st.markdown("#### Which corrective actions really stick?")
    a = df.dropna(subset=["corrective_action", "action_effective"]).groupby(["category", "corrective_action"]).agg(cases=("complaint_id", "count"), effective_pct=("action_effective", lambda x: round(100 * (x == "Y").mean(), 0))).reset_index().sort_values("effective_pct")
    st.dataframe(a[a["cases"] >= 15].head(12), width="stretch", hide_index=True)
    st.caption("Lowest-effectiveness actions (e.g. retraining or counselling alone) are down-ranked by the recommender.")

elif page == "Relationships & Root Cause":
    st.subheader("4 · Relationships between product, customer, complaint type and root cause")
    dim = st.radio("Show lift for", ["product", "customer", "line", "customer_type"], horizontal=True, format_func=lambda x: x.replace("_", " ").title())
    lift, ct = E.lift_matrix(df, dim)
    f = px.imshow(lift, color_continuous_scale=HEAT, zmin=0, zmax=3, aspect="auto", title="Lift: how much more often this category occurs for this item than average (1.0 = average)")
    f.update_traces(text=lift.values, texttemplate="%{text}", textfont_size=9); st.plotly_chart(fig_style(f, 440 if dim != "customer" else 560), width="stretch")
    c1, c2 = st.columns(2)
    rc = E.root_cause_table(df); f = px.imshow(rc, color_continuous_scale=HEAT, aspect="auto", title="Root cause share within each category (known root causes only)")
    f.update_traces(text=(rc * 100).round(0).astype(int).astype(str) + "%", texttemplate="%{text}", textfont_size=10); c1.plotly_chart(fig_style(f, 380), width="stretch")
    c2.markdown("**Strongest association rules**  (antecedent → category)"); c2.dataframe(B["rules"].head(14), width="stretch", hide_index=True, height=380)
    st.caption(f"Root cause recorded for {df['root_cause'].notna().mean():.0%} of complaints – older records have more gaps, which the system tolerates and flags for back-filling.")
    st.markdown("#### Process evidence from batch records")
    var = st.selectbox("Process / test parameter", ["seal_temp_c", "moisture_pct", "fill_weight_dev_g"], format_func=lambda x: {"seal_temp_c": "Seal temperature (°C)", "moisture_pct": "Moisture (%)", "fill_weight_dev_g": "Fill-weight deviation (g)"}[x])
    st.plotly_chart(fig_style(px.box(df.dropna(subset=[var]), x="category", y=var, color_discrete_sequence=[OR], title=f"{var} recorded on complaint batches, by complaint category"), 340), width="stretch")

elif page == "Risk & Prediction":
    st.subheader(f"5 · High-risk products/processes and 6 · Predicted complaint risk for {next_month}")
    s = B["scores"]; t = s.copy(); t["risk"] = t["risk"].round(3)
    c1, c2 = st.columns([1.2, 1])
    c1.markdown("**Risk league table** (probability of ≥1 complaint next month, with the top reasons)"); c1.dataframe(t.head(20), width="stretch", hide_index=True, height=480, column_config={"why": st.column_config.TextColumn("why (top drivers)", width="large"), "risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.2f")})
    pr = s.groupby("product")["risk"].agg(["max", "sum"]).rename(columns={"max": "highest_category_risk", "sum": "expected_complaint_categories"}).sort_values("highest_category_risk", ascending=False).round(2)
    c2.markdown("**Product ranking**"); c2.dataframe(pr, width="stretch", height=480)
    r = rm.report; st.markdown("#### Does the model really work? Back-test on unseen months")
    st.caption(f"Trained on data up to Dec 2023, tested on {r['test_months']} later months ({r['n_test']:,} product × category × month cases, {r['base_rate']:.0%} had a complaint).")
    a = st.columns(5); a[0].metric("ROC-AUC", r["roc_auc"]); a[1].metric("PR-AUC (model)", r["pr_auc"], f"{r['pr_auc']-r['pr_auc_baseline']:+.3f} vs simple baseline")
    a[2].metric("Lift in top 10 % risk", f"{r['lift_top10pct']}×"); a[3].metric("Recall in top 10 %", f"{r['recall_top10pct']:.0%}"); a[4].metric("Precision of top-10 alerts / month", f"{r['precision_at10_model']:.0%}", f"{(r['precision_at10_model']-r['precision_at10_baseline'])*100:+.0f} pts vs baseline")
    st.caption("Baseline = rank pairs by their complaints in the last 6 months. With synthetic data most variation is random noise, so gains over the baseline are modest – this table is how the claim will be verified on the real data.")
    meta, p, y = rm.test_pred; bins = pd.qcut(p, 10, duplicates="drop"); cal = pd.DataFrame({"pred": p, "actual": y, "bin": bins}).groupby("bin", observed=True).mean().reset_index(drop=True)
    f = go.Figure(); f.add_scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", line=dict(color=OR), name="model"); f.add_scatter(x=[0, .6], y=[0, .6], line=dict(color=GR, dash="dot"), name="perfect calibration")
    f.update_layout(title="Calibration: predicted vs actual complaint frequency (deciles)", xaxis_title="predicted risk", yaxis_title="actual frequency")
    g = B["importance"].head(8).iloc[::-1]; f2 = px.bar(x=g.values, y=g.index, orientation="h", color_discrete_sequence=[OR], title="What drives risk? (permutation importance)", labels={"x": "", "y": ""})
    d1, d2 = st.columns(2); d1.plotly_chart(fig_style(f, 330), width="stretch"); d2.plotly_chart(fig_style(f2, 330), width="stretch")

elif page == "Preventive Recommendations":
    st.subheader("7 · Preventive & corrective action recommender")
    c1, c2 = st.columns(2); p = c1.selectbox("Product", sorted(df["product"].unique())); s = B["scores"]
    top_cat = s[s["product"] == p].iloc[0]["category"]; cat = c2.selectbox("Complaint category", E.CATS, index=E.CATS.index(top_cat))
    txt = st.text_input("Optional: describe the new complaint / observation", placeholder="e.g. seal opening on arrival at distributor, batch B2610-B12")
    row = s[(s["product"] == p) & (s["category"] == cat)].iloc[0]; a, b = st.columns(2)
    a.metric(f"Predicted risk · {next_month}", f"{row['risk']:.0%}", str(row["tier"]), delta_color="off", delta_arrow="off"); b.markdown(f"**Why:** {row['why'] or '—'}")
    t, cases, checks = B["recommender"].recommend(p, cat, txt)
    st.markdown("#### Ranked actions (by proven effectiveness on similar past cases)"); st.dataframe(t, width="stretch", hide_index=True)
    st.markdown("#### Pre-dispatch checks (SOP rule layer)"); [st.markdown(f"- {c}") for c in checks]
    st.markdown("#### Most similar past cases (evidence)"); st.dataframe(cases, width="stretch", hide_index=True)

elif page == "Log New Complaint":
    st.subheader("Add a new complaint – the system learns from it")
    st.caption("The complaint text is auto-classified instantly; after saving, press **Retrain models** to refresh risk scores (automatic nightly in production).")
    with st.form("new"):
        c1, c2, c3 = st.columns(3); d = c1.date_input("Date", dt.date(2026, 11, 24)); cu = c2.selectbox("Customer", sorted(df["customer"].unique())); p = c3.selectbox("Product", sorted(df["product"].unique()))
        txt = st.text_area("Complaint description", placeholder="e.g. Customer found a plastic piece inside the pouch")
        c4, c5, c6 = st.columns(3); batch = c4.text_input("Batch number", "B2611-A01"); line = c5.selectbox("Line", [1, 2, 3, 4]); sev = c6.slider("Severity", 1, 5, 3)
        ok = st.form_submit_button("Classify & preview")
    if ok and txt.strip():
        pr = B["clf"].predict_proba(txt); st.session_state["pred"] = pr; st.session_state["rec"] = dict(date=str(d), customer=cu, product=p, complaint_text=txt, batch_no=batch, line=line, severity=sev)
    if "pred" in st.session_state:
        pr = st.session_state["pred"]; st.markdown(f"**Predicted category: {pr.index[0]}** ({pr.iloc[0]:.0%} confidence)")
        st.bar_chart(pr.head(5)); cat = st.selectbox("Confirm / override category", E.CATS, index=E.CATS.index(pr.index[0]))
        if st.button("💾 Save complaint to database"):
            rec = dict(st.session_state["rec"]); rec["category"] = cat; rec["customer_type"] = df.loc[df["customer"] == rec["customer"], "customer_type"].iloc[0]
            cid = E.add_complaint(rec); del st.session_state["pred"]; st.success(f"Saved as {cid}. Press 'Retrain models now' in the sidebar to update the analysis."); 

elif page == "KPIs: Road to Zero":
    st.subheader("9 · KPI scorecard – measurable steps towards Zero Customer Complaints")
    k = B["kpi"]; r = rm.report
    rows = [("Complaints per million units", k["ppm_12m"], f"-50 % → {k['ppm_12m']*.5:.1f}", k["ppm_12m"] <= k["ppm_prev12m"], "Complaints ÷ units dispatched × 10⁶ (last 12 months)"),
            ("Repeat-complaint rate", f"{k['repeat_rate']}%", "< 5 %", k["repeat_rate"] < 5, "% of complaints where the same customer already complained about the same product + category in the previous 12 months"),
            ("CAPA effectiveness", f"{k['capa_effective']}%", "> 90 %", k["capa_effective"] > 90, "% of actions recorded as effective (last 24 months)"),
            ("Average closure time (days)", k["avg_closure_days"], "≤ 7 days", k["avg_closure_days"] <= 7, "Receipt to verified closure"),
            ("Early-warning precision (top-10 alerts/month)", f"{r['precision_at10_model']:.0%}", "≥ 60 %", r["precision_at10_model"] >= .6, "Share of monthly top-10 risk alerts followed by a real complaint (back-test)"),
            ("Chronic problem areas", k["chronic_pairs"], "0", k["chronic_pairs"] == 0, "Product + category pairs with ≥ 10 complaints in 24 months"),
            ("Complaint-free product × category pairs (≥ 12 months)", f"{k['pairs_free_12m']} of {k['n_pairs']}", "Rising each quarter", False, "Pairs with no complaint for 12+ months – the footprint of 'zero complaints'")]
    t = pd.DataFrame(rows, columns=["KPI", "Current", "Target (proposed)", "On target?", "Definition"]); t["Current"] = t["Current"].astype(str); t["On target?"] = t["On target?"].map({True: "✅", False: "⚠️"})
    st.dataframe(t, width="stretch", hide_index=True, column_config={"KPI": st.column_config.TextColumn(width="large"), "Current": st.column_config.TextColumn(width="small"), "Target (proposed)": st.column_config.TextColumn(width="medium"), "On target?": st.column_config.TextColumn(width="small"), "Definition": st.column_config.TextColumn(width="large")})
    st.markdown("#### Longest complaint-free streaks (months since last complaint, product × category)")
    s = k["streaks"].head(15).rename("months").to_frame(); s["pair"] = s.index
    st.plotly_chart(fig_style(px.bar(s.iloc[::-1], x="months", y="pair", orientation="h", color_discrete_sequence=[GN], title="Longer = closer to zero"), 420), width="stretch")
    st.caption("Targets are proposals; baselines come from this database and should be agreed with ABT-CPD management.")

else:
    st.subheader("About & Method")
    st.markdown(f"""
**ZeroC-AI** turns 19 years of complaint history into predictions and preventive actions.

| Step | Method |
|---|---|
| Storage | SQLite / SQL database (`complaints`, `volumes`) – new complaints are appended from the form |
| Classification | TF-IDF + Logistic Regression on complaint text, 9 categories; k-means theme discovery |
| Pareto / recurrence | Counts + cumulative %, rolling 24-month recurrence, chronic flag |
| Relationships | Lift matrices, association rules (support / confidence / lift), root-cause shares (6M) |
| Early warning | u-chart (3σ) and CUSUM on complaints per 100 k units |
| Risk model | Gradient boosting on lagged counts, seasonality, volume; predicts P(complaint next month) per product × category; time-based back-test |
| Recommender | Similar-case retrieval + effectiveness-weighted action ranking + SOP rule checks |

**Data note:** this demo runs on a *synthetic* 19-year database with planted patterns (a chronic problem, a rising trend, a seasonal one). It proves the pipeline; accuracy numbers must be re-measured on the real ABT-CPD data. To load real data, put it in the same columns (see README) and rerun.
""")
