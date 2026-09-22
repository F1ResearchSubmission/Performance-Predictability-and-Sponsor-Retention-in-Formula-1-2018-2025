"""Robustness checks.

1. Sponsor matching: results under four identity rules. The headline uses
   normalized names plus 'typo' merges from sponsor_crosswalk.csv.
2. Record corrections: headline model with subsets of the six sourced
   corrections in data/sponsor_record_corrections.csv.
3. Listing gaps: share of recorded exits where the sponsor reappears with the
   same team two seasons later. This can be a listing gap OR a real return;
   it is not a measured false-exit rate.

Output: output/robustness.txt
"""

import importlib

import numpy as np
import pandas as pd

from common import OUTPUT, build_panel, load_sponsors, normalize, sponsor_key, with_reference

model = importlib.import_module("04_model")

RULES = {
    "Exact spelling": lambda name: name,
    "Normalized": normalize,
    "Normalized + typo (headline)": sponsor_key(("typo",)),
    "Normalized + typo + brand": sponsor_key(("typo", "brand")),
}


def summarize(panel):
    m = model.fit(panel)
    row = {"n": len(panel), "Continuation": panel["Retained"].mean()}
    for term, label in [("QualRaceCorrelation", "Predict."),
                        ("MeanRaceFinishPos", "Finish"),
                        ("C(Industry)[T.Automotive]", "Auto"),
                        ("C(Industry)[T.Crypto & Web3]", "Crypto")]:
        row[f"{label} OR"] = np.exp(m.params[term])
        row[f"{label} p"] = m.pvalues[term]
    row["AUC"] = model.held_out_auc(panel)
    return row


CORRECTION_SETS = {
    "None": [],
    "All except Volvo (S002-S006)": ["S002", "S003", "S004", "S005", "S006"],
    "Volvo only (S001)": ["S001"],
    "All six (headline)": None,
}


def reappearance(sp):
    by = {k: set(g["Key"]) for k, g in sp.groupby(["Team", "Season"])}
    back = []
    for (team, year), keys in by.items():
        if (team, year + 2) not in by:
            continue
        for k in keys - by.get((team, year + 1), set()):
            back.append(k in by[(team, year + 2)])
    return np.mean(back), len(back)


def main():
    rows = {}
    for label, rule in RULES.items():
        rows[label] = summarize(with_reference(build_panel(load_sponsors(rule))))
    table = pd.DataFrame(rows).T.astype({"n": int})

    crows = {label: summarize(with_reference(build_panel(load_sponsors(correction_ids=ids))))
             for label, ids in CORRECTION_SETS.items()}
    ctable = pd.DataFrame(crows).T.astype({"n": int})

    rate, n = reappearance(load_sponsors())
    text = ("Sponsor-matching sensitivity AFTER source-record corrections; p-values use CR1/t(9).\n"
            + table.to_string(float_format=lambda v: f"{v:.3f}")
            + "\n\nSensitivity to the sourced record corrections (normalized + typo matching; CR1/t(9)).\n"
            + ctable.to_string(float_format=lambda v: f"{v:.3f}")
            + f"\n\nRecorded exits that reappear two seasons later: {rate:.1%} of {n:,}\n"
            + "These are potential listing gaps or genuine returning partnerships, not verified errors.\n")
    print(text)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "robustness.txt").write_text(text)


if __name__ == "__main__":
    main()
