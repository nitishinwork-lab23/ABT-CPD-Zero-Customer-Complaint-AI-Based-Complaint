"""Quick self-test: python test_smoke.py"""
import engine as E, pandas as pd
df, vol = E.load(); assert len(df) > 1000
clf, rep = E.nlp_report(df); assert rep["macro_f1"] > .6, rep
assert E.pareto(df, "category")["cum_%"].iloc[-1] > 99
assert (E.recurrence(df)["status"] == "Chronic").any()
rm = E.RiskModel(df, vol); r = rm.report; assert r["roc_auc"] > .6, r
s = rm.score_next(); assert s["risk"].between(0, 1).all() and len(s) == rm.P * rm.C
t, cases, checks = E.Recommender(df).recommend("Product B", "Foreign matter"); assert len(t) >= 3
print("ALL TESTS PASSED", {k: r[k] for k in ("roc_auc", "pr_auc", "precision_at10_model")})
