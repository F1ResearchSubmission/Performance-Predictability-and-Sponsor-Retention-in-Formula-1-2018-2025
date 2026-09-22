"""Aggregate 2018-2024 driver results to team-race and team-season level.

Input:  data/raw/f1_performance_data_2018_2025_complete.csv
Output: data/processed/f1_team_race_level_2018_2024.csv
        data/processed/f1_team_season_metrics_2018_2024.csv
All 2025 performance is quarantined; 2025 sponsor listings remain outcomes.

Team metrics average both drivers. Rebranded teams are treated as one
continuous entity (19 names -> 10 constructors).
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import PROCESSED, RAW, DATA, METRICS

INPUT = RAW / "f1_performance_data_2018_2025_complete.csv"

TEAM_NAME_MAP = {
    "Red Bull Racing": "Red Bull Racing",
    "Mercedes": "Mercedes",
    "Ferrari": "Ferrari",
    "McLaren": "McLaren",
    "Alpine": "Alpine/Renault",
    "Renault": "Alpine/Renault",
    "Aston Martin": "Aston Martin/Racing Point/Force India",
    "Racing Point": "Aston Martin/Racing Point/Force India",
    "Force India": "Aston Martin/Racing Point/Force India",
    "RB": "RB/AlphaTauri/Toro Rosso",
    "Racing Bulls": "RB/AlphaTauri/Toro Rosso",
    "AlphaTauri": "RB/AlphaTauri/Toro Rosso",
    "Toro Rosso": "RB/AlphaTauri/Toro Rosso",
    "Kick Sauber": "Sauber/Alfa Romeo",
    "Alfa Romeo": "Sauber/Alfa Romeo",
    "Alfa Romeo Racing": "Sauber/Alfa Romeo",
    "Sauber": "Sauber/Alfa Romeo",
    "Haas F1 Team": "Haas",
    "Williams": "Williams",
}


def standardize_team_names(df):
    unmapped = set(df["Team"]) - set(TEAM_NAME_MAP)
    if unmapped:
        raise ValueError(f"Team names missing from TEAM_NAME_MAP: {sorted(unmapped)}")
    df["TeamStandardized"] = df["Team"].map(TEAM_NAME_MAP)
    return df


def aggregate_to_team_race_level(df):
    # Means use whichever drivers have a valid position (7 rows have one missing).
    team_race = df.groupby(
        ["Season", "RoundNumber", "RaceName", "TeamStandardized"]
    ).agg(
        AvgQualifyingPos=("QualifyingPos", "mean"),
        AvgRaceFinishPos=("RaceFinishPos", "mean"),
        TotalRacePoints=("RacePoints", "sum"),
        NumDrivers=("Driver", "count"),
    ).reset_index()
    return team_race.rename(columns={"TeamStandardized": "Team"})


def compute_season_metrics(team_race):
    rows = []
    for (season, team), g in team_race.groupby(["Season", "Team"]):
        quali, race = g["AvgQualifyingPos"].dropna(), g["AvgRaceFinishPos"].dropna()
        if len(quali) < 3 or len(race) < 3:
            continue

        paired = g.dropna(subset=["AvgQualifyingPos", "AvgRaceFinishPos"])
        corr, p = (stats.pearsonr(paired["AvgQualifyingPos"], paired["AvgRaceFinishPos"])
                   if len(paired) >= 3 else (np.nan, np.nan))
        # Positive = places gained from qualifying to finish.
        change = paired["AvgQualifyingPos"] - paired["AvgRaceFinishPos"]

        rows.append({
            "Season": season,
            "Team": team,
            "NumRaces": len(g),
            "MeanQualifyingPos": round(quali.mean(), 2),
            "MeanRaceFinishPos": round(race.mean(), 2),
            "StdQualifyingPos": round(quali.std(), 2),
            "StdRaceFinishPos": round(race.std(), 2),
            "QualRaceCorrelation": round(corr, 4) if not np.isnan(corr) else np.nan,
            "QualRaceCorr_pValue": round(p, 4) if not np.isnan(p) else np.nan,
            "PositionChange_Mean": round(change.mean(), 2),
            "PositionChange_Std": round(change.std(), 2),
            "TotalPoints": g["TotalRacePoints"].sum(),
            "PointsPerRace": round(g["TotalRacePoints"].mean(), 2),
        })

    return pd.DataFrame(rows).sort_values(["Season", "MeanRaceFinishPos"], ignore_index=True)


if __name__ == "__main__":
    original = pd.read_csv(INPUT)
    quarantine = DATA / "quarantine"
    quarantine.mkdir(exist_ok=True)
    original.loc[original.Season == 2025].to_csv(quarantine / "performance_2025_not_used.csv", index=False)
    df = standardize_team_names(original.loc[original.Season.between(2018, 2024)].copy())
    expected = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24}
    if df.groupby("Season").RoundNumber.nunique().to_dict() != expected:
        raise ValueError("Unexpected race coverage")
    if df.duplicated(["Season", "RoundNumber", "Driver"]).any():
        raise ValueError("Duplicate driver-event record")
    team_race = aggregate_to_team_race_level(df)
    season = compute_season_metrics(team_race)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    team_race.to_csv(PROCESSED / "f1_team_race_level_2018_2024.csv", index=False)
    season.to_csv(METRICS, index=False)
    print(f"{len(team_race):,} team-race rows, {len(season)} team-season rows")
