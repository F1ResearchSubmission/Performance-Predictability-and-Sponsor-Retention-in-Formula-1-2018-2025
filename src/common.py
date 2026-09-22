"""Shared paths, loaders, and sponsor-identity rules."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
METRICS = PROCESSED / "f1_team_season_metrics_2018_2024.csv"

# Sponsor file uses short team names; performance files use the long form.
TEAM_MAP = {
    "Aston Martin": "Aston Martin/Racing Point/Force India",
    "RB/AlphaTauri": "RB/AlphaTauri/Toro Rosso",
    "Red Bull": "Red Bull Racing",
}


def normalize(name):
    """Lowercase, letters and digits only. 'Ray-Ban' and 'Ray Ban' -> 'rayban'."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def sponsor_key(decisions=("typo",)):
    """Normalized name, then crosswalk merges of the given decision types."""
    cw = pd.read_csv(DATA / "sponsor_crosswalk.csv")
    merges = {normalize(v): normalize(c)
              for v, c, d in zip(cw.Variant, cw.Canonical, cw.Decision)
              if d in decisions}
    return lambda name: merges.get(normalize(name), normalize(name))


def corrected_sponsor_records(correction_ids=None):
    """Apply narrowly scoped, source-backed edits without overwriting raw data.

    correction_ids: subset of correction IDs to apply; None applies all.
    """
    sp = pd.read_csv(RAW / "f1_sponsors_standardized.csv")
    sp["OriginalSponsorName"] = sp["SponsorName"]
    sp["CorrectionID"] = ""
    edits = pd.read_csv(DATA / "sponsor_record_corrections.csv").fillna("")
    if correction_ids is not None:
        edits = edits[edits.ID.isin(correction_ids)]
    for e in edits.itertuples(index=False):
        mask = (sp.Team == e.Team) & (sp.Season == e.Season) & (sp.SponsorName == e.OldName)
        if e.Action == "rename":
            if mask.sum() != 1:
                raise ValueError(f"Correction {e.ID}: expected exactly one source row")
            sp.loc[mask, "SponsorName"] = e.NewName
            sp.loc[mask, "CorrectionID"] = e.ID
        elif e.Action == "add":
            if ((sp.Team == e.Team) & (sp.Season == e.Season) & (sp.SponsorName == e.NewName)).any():
                raise ValueError(f"Correction {e.ID}: addition already present")
            sp = pd.concat([sp, pd.DataFrame([dict(Team=e.Team, Season=e.Season,
                SponsorName=e.NewName, SponsorIndustry=e.Industry,
                OriginalSponsorName="", CorrectionID=e.ID)])], ignore_index=True)
        else:
            raise ValueError(f"Unknown correction action: {e.Action}")
    # The two 2020 BlueDEF fragments intentionally become one source record.
    merged = sp.duplicated(["Team", "Season", "SponsorName"], keep=False)
    if merged.any():
        groups = sp.loc[merged].groupby(["Team", "Season", "SponsorName"])
        for (team, year, name), g in groups:
            if (team, year, name) != ("Haas", 2020, "BlueDEF") or len(g) != 2:
                raise ValueError("Unexpected duplicate introduced by source corrections")
            first = g.index[0]
            sp.loc[first, "OriginalSponsorName"] = " | ".join(g.OriginalSponsorName)
            sp.loc[first, "CorrectionID"] = " | ".join(g.CorrectionID)
    return sp.drop_duplicates(["Team", "Season", "SponsorName"]).reset_index(drop=True)


def load_sponsors(key=None, correction_ids=None):
    """One row per sponsor-team-season, with a single industry per sponsor."""
    sp = corrected_sponsor_records(correction_ids)
    sp["Team"] = sp["Team"].replace(TEAM_MAP)

    tax = pd.read_csv(DATA / "taxonomy_map.csv")
    sp["Industry"] = sp["SponsorIndustry"].map(dict(zip(tax.IndustryRaw, tax.Industry)))
    unmapped = sp.loc[sp["Industry"].isna(), "SponsorIndustry"].unique()
    if len(unmapped):
        raise ValueError(f"Industry labels missing from taxonomy_map.csv: {unmapped}")

    sp["Key"] = sp["SponsorName"].map(key or sponsor_key())

    # Most frequent label per sponsor; ties go to the earliest recorded label.
    sp = sp.sort_values(["Season", "Team"], kind="stable")
    modal = sp.groupby("Key")["Industry"].agg(lambda v: v.value_counts(sort=False).idxmax())
    sp["Industry"] = sp["Key"].map(modal)

    teams = set(sp["Team"])
    metrics_teams = set(pd.read_csv(METRICS)["Team"])
    assert teams == metrics_teams, f"Team names do not align: {teams ^ metrics_teams}"

    return sp.drop_duplicates(["Team", "Season", "Key"]).reset_index(drop=True)


def build_panel(sp):
    """Sponsor-season rows where next season is observed (2018-2024 origins)."""
    by = {k: set(g["Key"]) for k, g in sp.groupby(["Team", "Season"])}
    first_season = sp.groupby("Team")["Season"].min().to_dict()

    rows = []
    for (team, year), g in sp.groupby(["Team", "Season"]):
        if (team, year + 1) not in by:
            continue
        for _, r in g.iterrows():
            tenure, back = 1, year - 1
            while (team, back) in by and r.Key in by[(team, back)]:
                tenure, back = tenure + 1, back - 1
            rows.append({
                "Team": team,
                "Season": year,
                "Key": r.Key,
                "SponsorName": r.SponsorName,
                "Industry": r.Industry,
                "Retained": int(r.Key in by[(team, year + 1)]),
                "Tenure": tenure,
                # Tenure reaching back to 2018 is a lower bound.
                "LeftCensored": int(year - tenure + 1 == first_season[team]),
                "TeamSponsorCount": len(g),
            })

    panel = pd.DataFrame(rows)
    metrics = pd.read_csv(METRICS)
    cols = ["Team", "Season", "MeanRaceFinishPos", "MeanQualifyingPos",
            "QualRaceCorrelation", "PositionChange_Std", "PointsPerRace"]
    n = len(panel)
    panel = panel.merge(metrics[cols], on=["Team", "Season"], how="left", validate="many_to_one")
    if len(panel) != n or panel[cols].isna().any().any():
        raise ValueError("Missing performance covariate or invalid merge")
    if panel.duplicated(["Team", "Season", "Key"]).any():
        raise ValueError("Duplicate analysis observation")

    panel["LogTenure"] = np.log(panel["Tenure"])
    # Negated so higher = more predictable, matching QualRaceCorrelation.
    panel["PredStability"] = -panel["PositionChange_Std"]
    panel["SeasonCentered"] = panel["Season"] - 2021
    return panel


def with_reference(panel, ref="Technology"):
    out = panel.copy()
    out["Industry"] = pd.Categorical(
        out["Industry"], categories=[ref] + sorted(set(out["Industry"]) - {ref}))
    return out
