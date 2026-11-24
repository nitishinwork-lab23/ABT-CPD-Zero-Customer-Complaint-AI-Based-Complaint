"""Synthetic 19-year complaint database for the ZeroC-AI prototype.

IMPORTANT: this is SYNTHETIC data with planted patterns so the pipeline can be demonstrated
before the organisers' real ABT-CPD database is loaded. Replace with real data via `load_real_data.py`
(same column names) - nothing else in the app changes.
"""
import sqlite3, sys, os
import numpy as np, pandas as pd

SEED = 36
START, END = "2007-11-01", "2026-10-31"      # ~19 years
PRODUCTS = [f"Product {c}" for c in "ABCDEFGHIJKL"]
CUST_TYPES = ["Retail chain", "Distributor", "Export", "Institutional"]
CATS = ["Quality deviation", "Physical defect", "Packaging", "Foreign matter", "Infestation",
        "Sensory", "Weight/Quantity", "Process defect", "Other"]
ROOTS = ["Man", "Machine", "Material", "Method", "Measurement", "Environment"]
ROOT_BY_CAT = {
 "Foreign matter": [.05, .40, .35, .05, .05, .10], "Infestation": [.05, .05, .25, .05, .00, .60],
 "Packaging":      [.10, .60, .15, .05, .05, .05], "Sensory":     [.05, .05, .50, .30, .05, .05],
 "Weight/Quantity":[.10, .45, .00, .05, .40, .00], "Quality deviation": [.10, .10, .30, .40, .10, .00],
 "Physical defect":[.30, .40, .10, .10, .00, .10], "Process defect": [.30, .10, .00, .50, .10, .00],
 "Other":          [.30, .00, .10, .30, .00, .30]}
ACTIONS = {
 "Foreign matter": [("Verify metal detector & rejection test every shift", .85), ("Sieve / magnet integrity check", .75),
                    ("Supplier raw-material audit", .70), ("Retrain operators on GMP", .45), ("Counselling only - no systemic action", .20)],
 "Infestation":    [("Revise fumigation & pest-control schedule", .80), ("FIFO + warehouse humidity control", .75),
                    ("Hold packing moisture within spec", .80), ("Retrain operators on GMP", .40), ("Distributor warehouse inspection", .55)],
 "Packaging":      [("Recalibrate sealer temperature / pressure", .85), ("Replace worn sealing jaw", .80),
                    ("Pack-integrity (leak) test 1-in-10", .70), ("Review film supplier / spec", .65), ("Retrain operators", .40)],
 "Sensory":        [("Sensory approval of raw material at receipt", .80), ("Revise roasting / process time-temperature", .75),
                    ("Shelf-life & storage-condition review", .60), ("Change supplier", .55), ("Retrain operators", .35)],
 "Weight/Quantity":[("Recalibrate filler & check-weigher", .90), ("Net-content SPC at the line", .80),
                    ("Increase check-weigher verification frequency", .75), ("Review fill set-point per SKU", .70), ("Retrain operators", .40)],
 "Quality deviation": [("Tighten in-process QC limits", .75), ("Revise recipe / process SOP", .70),
                    ("Revise raw-material specification", .65), ("Lab instrument calibration check", .60), ("Retrain operators", .40)],
 "Physical defect":[("Preventive maintenance of machine", .80), ("Redesign shipper / secondary pack", .65),
                    ("Adjust line speed", .60), ("Visual inspection at pack-out", .60), ("Retrain operators on handling", .50)],
 "Process defect": [("Lock process parameters + alarm", .85), ("SOP revision & validation", .75),
                    ("Equipment recalibration", .75), ("Line audit", .60), ("Retrain operators", .50)],
 "Other":          [("Customer-specific review & agreement", .70), ("Documentation / labelling check", .65),
                    ("Transport & handling review", .60), ("Retrain staff", .40), ("No action - isolated case", .30)]}
OBJ = ["metal piece", "plastic fragment", "hair strand", "small stone", "wood splinter", "loose thread", "glass shard", "insect part"]
TEXT = {
 "Foreign matter": ["Found {o} inside the pack of {p}", "Customer reported a {o} in the product, batch {b}", "{o} observed in product while opening the pack",
                    "Foreign particle ({o}) seen in {p}", "Contamination complaint: {o} present in sachet"],
 "Infestation": ["Live insects found in the pack of {p}", "Weevils observed in the product, batch {b}", "Larvae and webbing seen inside the pouch",
                 "Mite infestation noticed on opening {p}", "Insect infestation reported at the customer warehouse for {p}"],
 "Packaging": ["Pouch seal leaking on arrival", "Seal open and product spilled in carton", "Pack torn at the seal, batch {b}",
               "Batch code printing smudged on {p}", "Outer carton crushed and pack damaged", "Wrong or missing batch code on pack"],
 "Sensory": ["Off smell reported in {p}", "Product tastes bitter, batch {b}", "Colour darker than usual", "Stale rancid odour on opening",
             "Flavour weak compared to previous lot", "Customer says aroma is not as usual"],
 "Weight/Quantity": ["Net weight less than declared on pack", "Pack underweight by {g} g, batch {b}", "Short quantity received in carton",
                     "Fill weight below label claim for {p}", "Overfilled packs causing seal problem"],
 "Quality deviation": ["Moisture high and lumps formed in {p}", "Particle size not as per specification", "Product not as per agreed specification, batch {b}",
                       "Texture inconsistent across packs", "Quality parameter out of limit in customer lab test"],
 "Physical defect": ["Broken and crushed pieces in pack", "Hard caked lumps found in {p}", "Discoloured patches visible on product",
                     "Dented damaged pack received", "Product powdered and fragmented in transit"],
 "Process defect": ["Mix ratio appears wrong in {p}, batch {b}", "Under-processed product, batch {b}", "Mixing not uniform across the pack",
                    "Label applied on wrong product", "Manufacturing date missing on pack"],
 "Other": ["Delay in delivery against purchase order", "Invoice quantity mismatch", "Wrong product supplied against order",
           "Short shelf life remaining at receipt", "Customer service response was slow"]}

def month_index(ts): return (ts.year - 2007) * 12 + ts.month - 11

def build(seed=SEED):
    rng = np.random.default_rng(seed)
    months = pd.period_range(START, END, freq="M"); T = len(months)
    P, C = len(PRODUCTS), len(CATS)
    # volume (units dispatched) per product-month
    pscale = rng.uniform(80_000, 380_000, P)
    t = np.arange(T)
    vol = np.zeros((P, T))
    for p in range(P):
        season = 1 + .12 * np.sin(2 * np.pi * (months.month.values - 3) / 12)
        vol[p] = pscale[p] * (0.7 + 0.5 * t / T) * season * rng.normal(1, .06, T)
    vols = pd.DataFrame([(PRODUCTS[p], str(months[i]), int(vol[p, i])) for p in range(P) for i in range(T)],
                        columns=["product", "month", "units_dispatched"])
    # base propensity
    w = rng.dirichlet(np.ones(C) * 1.2, P)
    w[:, CATS.index("Other")] *= .5
    for pp, cn in [(1, "Foreign matter"), (3, "Packaging"), (4, "Infestation"), (6, "Sensory"), (9, "Weight/Quantity")]:   # planted problem pairs
        w[pp, CATS.index(cn)] = max(w[pp, CATS.index(cn)], .22)
    w /= w.sum(1, keepdims=True)
    base = rng.uniform(.9, 1.6, P) * 1.15
    lam = np.zeros((P, C, T))
    yr = months.year.values + (months.month.values - 1) / 12
    improve = np.interp(yr, [2007, 2026.9], [1.45, .75])
    ci = {c: i for i, c in enumerate(CATS)}
    for p in range(P):
        for c in range(C):
            lam[p, c] = base[p] * w[p, c] * (vol[p] / 100_000) * improve * 0.55
    seasonal = {"Infestation": lambda m: 1 + 1.6 * ((m >= 6) & (m <= 10)), "Sensory": lambda m: 1 + .5 * ((m >= 3) & (m <= 5)),
                "Packaging": lambda m: 1 + .3 * ((m == 12) | (m <= 2))}
    for cn, f in seasonal.items(): lam[:, ci[cn], :] *= f(months.month.values)[None, :]
    # planted patterns
    lam[1, ci["Foreign matter"]] *= 3.5                                  # Product B chronic foreign matter
    lam[4, ci["Infestation"]] *= 2.2                                     # Product E seasonal infestation
    rise = np.interp(yr, [2007, 2023.0, 2026.9], [1, 1, 3.6]); lam[3, ci["Packaging"]] *= rise   # Product D rising packaging
    lam[6, ci["Sensory"]] *= 2.3                                         # Product G sensory (customer 07)
    lam[9, ci["Weight/Quantity"]] *= 2.6                                 # Product J weight (line 3)
    N = rng.poisson(lam)
    # customers
    ncust = 16; cust = [f"Customer {i:02d}" for i in range(1, ncust + 1)]
    ctype = {c: CUST_TYPES[i % 4] for i, c in enumerate(cust)}
    cw = rng.dirichlet(np.ones(ncust) * 2.0)
    rows = []; bid = 0
    for p in range(P):
        for c in range(C):
            for ti in np.nonzero(N[p, c])[0]:
                for _ in range(N[p, c, ti]):
                    d = months[ti].to_timestamp() + pd.Timedelta(days=int(rng.integers(0, months[ti].days_in_month)))
                    cat = CATS[c]
                    cw_ = cw.copy()
                    if cat == "Sensory" and p == 6: cw_[6] *= 8
                    if cat == "Packaging" and p == 3: cw_[2] *= 3
                    cw_ /= cw_.sum()
                    cu = cust[rng.choice(ncust, p=cw_)]
                    if rng.random() < .03: cat_lab = CATS[rng.integers(C)]
                    else: cat_lab = cat
                    line = int(rng.choice([1, 2, 3, 4], p=[.25, .25, .25, .25] if not (p == 9 and cat == "Weight/Quantity") else [.1, .1, .7, .1]))
                    batch = f"B{d.year % 100:02d}{d.month:02d}-{PRODUCTS[p][-1]}{rng.integers(1, 40):02d}"
                    tx = rng.choice(TEXT[cat]).format(o=rng.choice(OBJ), p=PRODUCTS[p], b=batch, g=int(rng.integers(5, 40)))
                    if rng.random() < .08:                      # noisy / mixed text
                        tx += ". Also " + rng.choice(TEXT[CATS[rng.integers(C)]]).format(o=rng.choice(OBJ), p=PRODUCTS[p], b=batch, g=10).lower()
                    if rng.random() < .15: tx = tx.replace("Customer", "Cust.").replace("reported", "rep.")
                    # root cause / action (missing more often in older years)
                    miss_r = np.interp(d.year, [2007, 2026], [.6, .12]); miss_a = np.interp(d.year, [2007, 2026], [.45, .05])
                    root = ROOTS[rng.choice(6, p=ROOT_BY_CAT[cat])] if rng.random() > miss_r else None
                    acts = ACTIONS[cat]
                    if rng.random() > miss_a:
                        ai = rng.choice(len(acts), p=np.array([.30, .22, .18, .18, .12]) if True else None)
                        action, pe = acts[ai]
                        eff = "Y" if rng.random() < pe else "N"
                        if (p == 1 and cat == "Foreign matter") or (p == 3 and cat == "Packaging" and d.year >= 2023):
                            eff = "Y" if rng.random() < pe * .55 else "N"     # chronic pairs: actions often do not stick
                    else: action, eff = None, None
                    closure = int(np.clip(rng.lognormal(np.interp(d.year, [2007, 2026], [3.0, 2.2]), .5), 1, 120))
                    sev = int(np.clip(rng.normal(2.8 if cat in ("Foreign matter", "Infestation") else 2.2, .9), 1, 5))
                    seal = round(float(rng.normal(168 + (9 if cat == "Packaging" else 0), 4)), 1) if d.year >= 2012 and rng.random() < .7 else None
                    moist = round(float(rng.normal(8.4 + (1.6 if cat in ("Infestation", "Quality deviation") else 0), .7)), 2) if d.year >= 2012 and rng.random() < .7 else None
                    fillw = round(float(rng.normal(-4.5 if cat == "Weight/Quantity" else 0.4, 2.0)), 1) if d.year >= 2012 and rng.random() < .7 else None
                    test = None
                    if d.year >= 2012 and rng.random() < .6: test = rng.choice(["Pass", "Marginal", "Fail"], p=[.45, .3, .25] if cat in ("Foreign matter", "Quality deviation", "Infestation") else [.7, .2, .1])
                    rows.append((d.strftime("%Y-%m-%d"), cu, ctype[cu], PRODUCTS[p], cat_lab, tx, batch, line, root, action, eff,
                                 closure, sev, seal, moist, fillw, test))
    df = pd.DataFrame(rows, columns=["date", "customer", "customer_type", "product", "category", "complaint_text", "batch_no", "line",
                                     "root_cause", "corrective_action", "action_effective", "closure_days", "severity",
                                     "seal_temp_c", "moisture_pct", "fill_weight_dev_g", "test_result"])
    df = df.sort_values("date").reset_index(drop=True); df.insert(0, "complaint_id", [f"C{i+1:05d}" for i in range(len(df))])
    return df, vols

def write_db(path, df, vols):
    if os.path.exists(path): os.remove(path)
    con = sqlite3.connect(path)
    df.to_sql("complaints", con, index=False); vols.to_sql("volumes", con, index=False)
    con.execute("CREATE INDEX ix_c_date ON complaints(date)"); con.commit(); con.close()

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "complaints.db")
    df, vols = build(); write_db(out, df, vols)
    print(f"wrote {len(df):,} complaints, {len(vols):,} volume rows -> {out}")
    print(df.category.value_counts().to_string())
