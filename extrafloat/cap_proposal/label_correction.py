"""
label_correction.py
===================
The outcome label used by the PD model and by the policy evidence (bad_state_3dpd_30d) is built from loan STATES, and a state exists only on a day a repayment is recorded
(build_loan_state.py). A loan with no repayment event after the due date therefore cannot show days_aging > 3 and is labelled good: non-payers, and loans partly repaid early
and then left, look safe. This script rebuilds the outcome from the raw repayments, independent of whether a repayment event exists, keeps every component visible, and reruns
the historical evidence with the corrected label, in this order:

  1. label versions and their components         2. burden tables (finer bands, confidence intervals)
  3. experience groups x burden                   4. reactivation regression
  5. PD target lineage

Versions of the label (all on the same labelled loans):
  A  original      bad_state_3dpd_30d as shipped (days past due above 3 on a recorded repayment state within 30 days of disbursement)
  B  A + no repayment at all in the 30-day window
  C  CORRECTED (primary): balance still outstanding at the end of the 4th day after the due date, i.e. more than 3 days past due, whether or not any repayment event exists
  D  strictest: not fully repaid within 30 days of disbursement

Writes aggregated tables to output\\label_audit\\ and a restricted loan-level file (phone numbers).   py\\python.exe analysis\\label_correction.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

pd.set_option("display.width", 230)
O = Path(r"F:\AGENT DATA\pipeline\output")
AUD = O / "label_audit"
AUD.mkdir(exist_ok=True)

# ---------------- repayments from the raw tables ----------------
con = duckdb.connect(r"F:\AGENT DATA\pipeline\pipeline.duckdb", read_only=True)
d = con.execute("select disbursement_fid fid, regexp_replace(msisdn,'\\D','','g') msisdn, cast(disbursement_ts as date) disb, cast(final_pay_date as date) due, "
                "disbursement_amount_ugx amt, loan_tenure_days tenure from stg_disbursements").fetchdf()
c = con.execute("select disbursement_fid fid, cast(repayment_ts as date) dt, repayment_amount_ugx pay from stg_collections").fetchdf()
d["disb"], d["due"], c["dt"] = pd.to_datetime(d.disb), pd.to_datetime(d.due), pd.to_datetime(c.dt)
c = c.merge(d[["fid", "disb", "due", "amt"]], on="fid").sort_values(["fid", "dt"])
c["cum"] = c.groupby("fid").pay.cumsum()


def cum_by(offset_col, days):
    cut = c[offset_col] + pd.Timedelta(days=days)
    return c[c.dt <= cut].groupby("fid").pay.sum()


D = d.set_index("fid")
D["paid_by_due4"] = cum_by("due", 4)
D["paid_30d"] = cum_by("disb", 30)
D["first_repay"] = c.groupby("fid").dt.min()
D["full_repay"] = c[c.cum >= c.amt - 0.5].groupby("fid").dt.min()
D = D.fillna({"paid_by_due4": 0, "paid_30d": 0}).reset_index()
D["fid"] = D.fid.astype(str)
D["days_to_full"] = (D.full_repay - D.disb).dt.days

# ---------------- labelled loans ----------------
L = pd.read_csv(O / "pd_training_dataset.csv", low_memory=False,
                usecols=["disbursement_fid", "msisdn", "loan_date", "split", "bad_state_3dpd_30d", "label_eligible_30d", "observed_prior_loan_count", "prior_severely_overdue_count",
                         "prior_late_fee_count", "consecutive_on_time_loans", "disbursement_amount_ugx"])
L["fid"] = L.disbursement_fid.astype(str).str.replace(r"\.0$", "", regex=True)
L["loan_date"] = pd.to_datetime(L.loan_date)
L = L.merge(D[["fid", "due", "amt", "tenure", "paid_by_due4", "paid_30d", "first_repay", "full_repay", "days_to_full"]], on="fid", how="left")
L["A"] = L.bad_state_3dpd_30d.astype(int)
L["no_repay_30d"] = (L.paid_30d <= 0).astype(int)
L["B"] = ((L.A == 1) | (L.no_repay_30d == 1)).astype(int)
L["C"] = (L.paid_by_due4 < L.amt - 0.5).astype(int)
L["D"] = (L.paid_30d < L.amt - 0.5).astype(int)
# audit components (mutually exclusive outcome status)
L["outcome_status"] = np.select(
    [L.no_repay_30d == 1, (L.C == 1) & (L.D == 1), (L.C == 1) & (L.D == 0), (L.A == 1) & (L.C == 0)],
    ["no repayment within the window", "not fully repaid within 30 days (past due 4+ days)", "repaid in full within 30 days but more than 3 days late", "observed delinquent state only"],
    "cured on time or within 3 days of the due date")
print(f"labelled loans {len(L):,} | train {int((L.split == 'train').sum()):,} | validation {int((L.split == 'validation').sum()):,}")
L.drop(columns=["disbursement_fid"]).to_csv(AUD / "loan_labels_corrected_RESTRICTED.csv", index=False)


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / np.maximum(n, 1)
    den = 1 + z * z / np.maximum(n, 1)
    ctr = (p + z * z / (2 * np.maximum(n, 1))) / den
    half = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z * z / (4 * np.maximum(n, 1) ** 2)) / den
    return ctr - half, ctr + half


# ============ 1. label versions ============
rows = []
for lab, mask in (("All labelled loans", L.split.notna()), ("Training (1 Jun 2025 - 31 Mar 2026)", L.split == "train"), ("Validation (1 Apr - 1 Aug 2026)", L.split == "validation")):
    m = L[mask]
    rows.append({"Population": lab, "Loans": len(m), "A original": m.A.mean(), "B + no repayment": m.B.mean(), "C corrected (primary)": m.C.mean(), "D not repaid in 30 days": m.D.mean(),
                 "Loans flipped good to bad by C": int(((m.A == 0) & (m.C == 1)).sum()), "Loans flipped bad to good by C": int(((m.A == 1) & (m.C == 0)).sum())})
V = pd.DataFrame(rows)
V.to_csv(AUD / "label_versions.csv", index=False)
print("\n[1] bad rate by label version:"); print(V.round(4).to_string(index=False))
st = L.outcome_status.value_counts().rename_axis("Outcome status").reset_index(name="Loans")
st["Share"] = st.Loans / len(L)
st["Original label"] = st["Outcome status"].map(L.groupby("outcome_status").A.mean())
st.to_csv(AUD / "outcome_status_components.csv", index=False)
print(st.round(4).to_string(index=False))

# ============ commission / burden for the 1 Mar - 1 Aug window (as in all earlier evidence) ============
mm = pd.read_csv(r"F:\AGENT DATA\mfs_daily_agent_mart_20260831.csv", usecols=["agent_msisdn", "commission"])
mm["msisdn"] = mm.agent_msisdn.astype(str).str.replace(r"\D", "", regex=True).astype("int64")
W = L[(L.loan_date >= "2026-03-01") & (L.loan_date <= "2026-08-01")].copy()
W["msisdn"] = W.msisdn.astype("int64")
W = W.merge(mm[["msisdn", "commission"]], on="msisdn", how="left")
W = W[W.commission > 0].copy()
W["burden"] = W.amt / (W.commission / 6)

# ============ 2. burden tables with confidence intervals ============
EDG = [0, 1, 1.5, 2, 2.5, 3, 4, 6, 1e9]
LAB = ["up to 1x", "1-1.5x", "1.5-2x", "2-2.5x", "2.5-3x", "3-4x", "4-6x", "over 6x"]
W["bband"] = pd.cut(W.burden, EDG, labels=LAB)


def band_table(df, by, order):
    out = []
    for k in order:
        g = df[df[by] == k]
        if len(g) == 0:
            continue
        lo, hi = wilson(g.C.sum(), len(g))
        out.append({"Band": k, "Loans": len(g), "Bad loans (corrected)": int(g.C.sum()), "Bad rate, corrected (C)": g.C.mean(), "95% low": float(lo), "95% high": float(hi),
                    "Bad rate, original (A)": g.A.mean(), "Bad rate, B": g.B.mean(), "Bad rate, D": g.D.mean()})
    return pd.DataFrame(out)


T2 = band_table(W, "bband", LAB)
T2.to_csv(AUD / "burden_bands_corrected.csv", index=False)
print("\n[2] burden bands (all labelled loans, 1 Mar - 1 Aug 2026):"); print(T2.round(4).to_string(index=False))

# ============ 3. experience groups x burden ============
n = W.observed_prior_loan_count.fillna(0).astype(int)
clean = (W.prior_severely_overdue_count.fillna(0) == 0) & (W.prior_late_fee_count.fillna(0) == 0) & (W.consecutive_on_time_loans.fillna(0) >= n)
sev = W.prior_severely_overdue_count.fillna(0) > 0
streak = W.consecutive_on_time_loans.fillna(0)
W["grp"] = np.select([n == 0, (n >= 1) & (n <= 2) & clean, (n >= 1) & (n <= 2), (n >= 3) & sev, (n >= 3) & (streak >= 3), (n >= 3)],
                     ["1 First loan", "2 One or two prior loans, all clean", "3 One or two prior loans, not clean", "4 Three or more, severe overdue on record",
                      "5 Three or more, on-time streak of 3+", "6 Three or more, streak under 3"], "other")
G = W.groupby("grp").agg(Loans=("C", "size"), Bad_C=("C", "sum"), Rate_C=("C", "mean"), Rate_A=("A", "mean")).reset_index()
G["95% low"], G["95% high"] = wilson(G.Bad_C, G.Loans)
G.to_csv(AUD / "experience_groups_corrected.csv", index=False)
print("\n[3a] experience groups, all burdens:"); print(G.round(4).to_string(index=False))
W["b5"] = pd.cut(W.burden, [0, 2, 3, 4, 6, 1e9], labels=["up to 2x", "2-3x", "3-4x", "4-6x", "over 6x"])
X = W[W.grp != "other"].groupby(["grp", "b5"], observed=True).agg(Loans=("C", "size"), Bad=("C", "sum"), Rate_C=("C", "mean"), Rate_A=("A", "mean")).reset_index()
X["95% low"], X["95% high"] = wilson(X.Bad, X.Loans)
X.to_csv(AUD / "experience_by_burden_corrected.csv", index=False)
print("\n[3b] corrected bad rate (C), experience group x burden:")
print(X.pivot(index="grp", columns="b5", values="Rate_C").round(4).to_string())
print("loans:"); print(X.pivot(index="grp", columns="b5", values="Loans").to_string())
print("original label (A):"); print(X.pivot(index="grp", columns="b5", values="Rate_A").round(4).to_string())

# ordering check within burden bands (first loan > 1-2 clean > 3+ streak)
ordc = []
for b in ["up to 2x", "2-3x", "3-4x", "4-6x"]:
    r = X[X.b5 == b].set_index("grp")
    try:
        f, e1, e3 = r.loc["1 First loan"], r.loc["2 One or two prior loans, all clean"], r.loc["5 Three or more, on-time streak of 3+"]
        ordc.append({"Burden": b, "First loan": f.Rate_C, "1-2 clean": e1.Rate_C, "3+ streak": e3.Rate_C, "Order holds (first > 1-2 clean > 3+ streak)": bool(f.Rate_C > e1.Rate_C > e3.Rate_C),
                     "First vs 1-2 clean: intervals overlap": bool(f["95% low"] <= e1["95% high"] and e1["95% low"] <= f["95% high"])})
    except KeyError:
        pass
O3 = pd.DataFrame(ordc)
O3.to_csv(AUD / "experience_ordering_check.csv", index=False)
print("\n[3c] ordering check:"); print(O3.round(4).to_string(index=False))

# ============ 4. reactivation regression (same specification as before) ============
reps = c[["fid", "dt"]].merge(d[["fid", "msisdn"]], on="fid")[["msisdn", "dt"]].drop_duplicates().sort_values("dt").rename(columns={"dt": "prev_rep"})
ld = d[["fid", "msisdn", "disb"]].sort_values("disb")
g = pd.merge_asof(ld, reps, left_on="disb", right_on="prev_rep", by="msisdn", direction="backward", allow_exact_matches=True)
g["gap"] = (g.disb - g.prev_rep).dt.days
g["fid"] = g.fid.astype(str)
R = W.merge(g[["fid", "gap"]], on="fid", how="left")
R = R[R.gap.notna()].copy()
R["gap_cat"] = pd.cut(R.gap, [-1, 30, 90, 150, 1e9], labels=["0-30", "31-90", "91-150", "over 150"])


def logit(M, ycol):
    cols = {"gap 31-90": (M.gap_cat == "31-90"), "gap 91-150": (M.gap_cat == "91-150"), "gap over 150": (M.gap_cat == "over 150"), "log burden": np.log(M.burden.clip(lower=0.05)),
            "log prior loans": np.log1p(M.observed_prior_loan_count.fillna(0)), "on-time streak (capped at 20)": M.consecutive_on_time_loans.fillna(0).clip(upper=20),
            "prior severe-overdue count": M.prior_severely_overdue_count.fillna(0).clip(upper=5), "log loan size": np.log(M.amt)}
    X_ = np.column_stack([np.ones(len(M))] + [np.asarray(v, float) for v in cols.values()])
    y = M[ycol].values.astype(float)
    b = np.zeros(X_.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X_ @ b))
        H = X_.T @ (X_ * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X_.shape[1])
        step = np.linalg.solve(H, X_.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    names = ["const"] + list(cols)
    return pd.DataFrame({"Odds ratio": np.exp(b), "Low (95%)": np.exp(b - 1.96 * se), "High (95%)": np.exp(b + 1.96 * se)}, index=names).drop(index="const")


Rb = R.dropna(subset=["burden"]).copy()
res = {}
for lab, ycol in (("Corrected (C)", "C"), ("Original (A)", "A"), ("B: A + no repayment", "B"), ("Strict (D)", "D")):
    res[lab] = logit(Rb, ycol)
OR = pd.concat({k: v.loc[["gap 31-90", "gap 91-150", "gap over 150"]] for k, v in res.items()}, axis=1)
OR.columns = [f"{a} - {b}" for a, b in OR.columns]
OR.reset_index().rename(columns={"index": "Term"}).to_csv(AUD / "reactivation_adjusted_corrected.csv", index=False)
print("\n[4a] adjusted odds ratios of a bad loan against a gap of 0-30 days (repeat loans, N=%d):" % len(Rb))
print(OR.round(2).to_string())
cnt = R.groupby("gap_cat", observed=True).agg(Loans=("C", "size"), Bad_C=("C", "sum"), Rate_C=("C", "mean"), Bad_A=("A", "sum"), Rate_A=("A", "mean")).reset_index()
cnt["95% low (C)"], cnt["95% high (C)"] = wilson(cnt.Bad_C, cnt.Loans)
cnt.to_csv(AUD / "reactivation_counts_corrected.csv", index=False)
print("\n[4b] loans and bad loans by gap band:"); print(cnt.round(4).to_string(index=False))

# standardised observed vs expected for the long-gap loans
R["prior_grp"] = pd.cut(R.observed_prior_loan_count.fillna(0), [-1, 2, 9, 1000], labels=["1-2 prior", "3-9 prior", "10+ prior"])
R["burden_grp"] = pd.cut(R.burden, [0, 2, 3, 4, 1e9], labels=["<=2x", "2-3x", "3-4x", ">4x"])
R["clean_grp"] = np.where(R.prior_severely_overdue_count.fillna(0) == 0, "no severe overdue before", "severe overdue before")
R["streak_grp"] = pd.cut(R.consecutive_on_time_loans.fillna(0), [-1, 0, 2, 5, 1000], labels=["0", "1-2", "3-5", "6+"])
keys = ["prior_grp", "burden_grp", "clean_grp", "streak_grp"]
out = []
for ycol in ("C", "A"):
    base = R[R.gap <= 90].groupby(keys, observed=True)[ycol].agg(["mean", "size"]).rename(columns={"mean": "exp_rate", "size": "ref_n"}).reset_index()
    Z = R.merge(base, on=keys, how="left")
    Z = Z[Z.ref_n >= 30]
    for band, lo_, hi_ in (("91-150", 91, 150), ("over 150", 151, 10 ** 6)):
        q = Z[(Z.gap >= lo_) & (Z.gap <= hi_)]
        rng = np.random.default_rng(0)
        rr = [(lambda s: s[ycol].sum() / s.exp_rate.sum())(q.sample(len(q), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
        out.append({"Label": "Corrected (C)" if ycol == "C" else "Original (A)", "Gap band": band, "Loans": len(q), "Bad loans": int(q[ycol].sum()), "Expected bad loans": float(q.exp_rate.sum()),
                    "Observed / expected": float(q[ycol].sum() / q.exp_rate.sum()), "Bootstrap low (95%)": float(np.percentile(rr, 2.5)), "Bootstrap high (95%)": float(np.percentile(rr, 97.5))})
SE = pd.DataFrame(out)
SE.to_csv(AUD / "reactivation_standardised_corrected.csv", index=False)
print("\n[4c] observed against expected (strata: prior loans x burden x prior overdue x streak; reference gap up to 90 days):"); print(SE.round(3).to_string(index=False))

# ============ 5. PD target lineage ============
print("\n[5] PD target lineage: raw repayments -> der_loan_state (one row per repayment date) -> bad_state_3dpd_30d (max days_aging within 30 days > 3) "
      "-> bad_state = (bad_state_3dpd_30d > 0) [pd_model/preprocessing/loan_history_features.py:193] -> TARGET_COL for XGBoost/LightGBM and the calibration map.")
lin = pd.DataFrame({"Step": ["Raw repayments (stg_collections)", "Loan states (der_loan_state): one row per repayment date plus the disbursement day", "bad_state_3dpd_30d: max days_aging on a state within 30 days > 3",
                             "bad_state = (bad_state_3dpd_30d > 0): the PD training target", "XGBoost / LightGBM trained on bad_state; isotonic calibration on bad_state rates"],
                    "Defect": ["", "No state on days without a repayment: a loan with no repayment after the due date has no late state", "Non-payers and loans left partly repaid are labelled good",
                               "Inherits the defect", "PD and calibration map learned from the defective label"]})
lin.to_csv(AUD / "pd_target_lineage.csv", index=False)
