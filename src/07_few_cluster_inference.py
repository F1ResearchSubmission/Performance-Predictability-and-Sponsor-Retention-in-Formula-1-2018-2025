"""Inference with 10 team clusters.

Conventional cluster-robust p-values can be too small when clusters are few.
Two checks on the headline coefficients:

1. Logit with CR1 standard errors and t(G-1) reference distribution.
2. Wild cluster restricted (WCR) bootstrap on the linear probability model,
   Webb six-point weights (recommended for fewer than ~12 clusters).
   MacKinnon, Nielsen & Webb (2023), "Cluster-robust inference: A guide to
   empirical practice", Journal of Econometrics.

Also reports the 95% CI for predictability, per unit and per SD. These are the p-values and CIs reported in the abstract.

Output: output/few_cluster_inference.txt
"""

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from common import OUTPUT, PROCESSED, with_reference

RHS = ("LogTenure + LeftCensored + C(Industry) + MeanRaceFinishPos"
       " + QualRaceCorrelation + SeasonCentered + TeamSponsorCount")
TERMS = {
    "QualRaceCorrelation": "Predictability",
    "MeanRaceFinishPos": "Mean race finish",
    "C(Industry)[T.Automotive]": "Automotive vs Technology",
    "C(Industry)[T.Crypto & Web3]": "Crypto & Web3 vs Technology",
}
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
REPS = 9999


def cr1_t(X, resid, groups, bread, k):
    """CR1 t-statistic numerator/denominator helper for coefficient k."""
    G, (N, K) = len(np.unique(groups)), X.shape
    meat = np.zeros((K, K))
    for g in np.unique(groups):
        s = X[groups == g].T @ resid[groups == g]
        meat += np.outer(s, s)
    V = (G / (G - 1)) * ((N - 1) / (N - K)) * bread @ meat @ bread
    return np.sqrt(V[k, k])


def wcr_pvalue(y, X, groups, k, rng):
    pinv = np.linalg.pinv(X)
    bread = pinv @ pinv.T
    beta = pinv @ y
    t_obs = beta[k] / cr1_t(X, y - X @ beta, groups, bread, k)

    Xr = np.delete(X, k, axis=1)
    fit_r = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    u_r = y - fit_r

    ids = {g: i for i, g in enumerate(np.unique(groups))}
    gi = np.array([ids[g] for g in groups])
    exceed = 0
    for _ in range(REPS):
        w = rng.choice(WEBB, size=len(ids))[gi]
        y_star = fit_r + w * u_r
        b = pinv @ y_star
        t = b[k] / cr1_t(X, y_star - X @ b, groups, bread, k)
        exceed += abs(t) >= abs(t_obs)
    return beta[k], (exceed + 1) / (REPS + 1)


def main():
    panel = with_reference(pd.read_csv(PROCESSED / "sponsor_panel.csv"))
    G = panel["Team"].nunique()
    lines = [f"Clusters: {G} teams, n = {len(panel):,}",
             "Logit CR1/t(9); WCR is a LINEAR PROBABILITY MODEL sensitivity check, not a logistic bootstrap.",
             "9999 restricted draws, Webb weights, seed 20260921; centered year and pseudoinverse.", ""]

    logit = smf.logit(f"Retained ~ {RHS}", panel).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": panel["Team"]}, use_t=True)
    y, X = patsy.dmatrices(f"Retained ~ {RHS}", panel, return_type="dataframe")
    names = list(X.columns)
    rng = np.random.default_rng(20260921)

    lines.append(f"{'':30s}{'Logit OR':>10s}{'p, t(G-1)':>12s}{'LPM coef':>11s}{'p, WCR boot':>13s}")
    boot_rows = []
    for term, label in TERMS.items():
        b, p_wcr = wcr_pvalue(y.to_numpy().ravel(), X.to_numpy(),
                              panel["Team"].to_numpy(), names.index(term), rng)
        lines.append(f"{label:30s}{np.exp(logit.params[term]):10.3f}"
                     f"{logit.pvalues[term]:12.4f}{b:11.4f}{p_wcr:13.4f}")
        boot_rows.append(dict(Model="primary",Term=term,LPM_coefficient=b,WCR_p=p_wcr,Draws=REPS))

    alt_rhs = RHS.replace("QualRaceCorrelation", "PredStability")
    ay, ax = patsy.dmatrices(f"Retained ~ {alt_rhs}", panel, return_type="dataframe")
    ab, ap = wcr_pvalue(ay.to_numpy().ravel(), ax.to_numpy(), panel.Team.to_numpy(),
                        list(ax.columns).index("PredStability"), rng)
    boot_rows.append(dict(Model="alternative",Term="PredStability",LPM_coefficient=ab,WCR_p=ap,Draws=REPS))
    lines.append(f"Alternative LPM: coefficient {ab:.4f}, WCR p {ap:.4f}")

    b, se = logit.params["QualRaceCorrelation"], logit.bse["QualRaceCorrelation"]
    crit = logit.t_test(np.eye(len(logit.params))[names.index("QualRaceCorrelation")]).conf_int()[0]
    sd = panel["QualRaceCorrelation"].std()
    lines += ["",
              "Predictability odds ratio, 95% CI (t(G-1))",
              f"  per 1.0 change in correlation   OR {np.exp(b):.3f}  [{np.exp(crit[0]):.3f}, {np.exp(crit[1]):.3f}]",
              f"  per 1 SD ({sd:.3f})               OR {np.exp(b * sd):.3f}  "
              f"[{np.exp(crit[0] * sd):.3f}, {np.exp(crit[1] * sd):.3f}]"]

    text = "\n".join(lines)
    print(text)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "few_cluster_inference.txt").write_text(text + "\n", encoding="utf-8")
    pd.DataFrame(boot_rows).to_csv(OUTPUT / "bootstrap_results.csv", index=False)


if __name__ == "__main__":
    main()
