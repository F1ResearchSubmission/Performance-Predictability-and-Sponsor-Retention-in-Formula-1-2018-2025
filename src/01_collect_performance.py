"""Download qualifying and race classifications, 2018-2025, via FastF1.

Output: data/raw/f1_performance_data_fastf1.csv
Writes to a separate file so the committed dataset is not overwritten; see
DATA_DICTIONARY.md for how the committed file differs.

Resumable: the F1 timing API is rate-limited (~500 calls/hour). If a limit is
hit, progress is saved and the script can be rerun after ~30 minutes.
Requires: pip install fastf1
"""

import os
import time
import warnings
import json

import fastf1
import pandas as pd

from common import RAW

SEASONS = range(2018, 2026)
EXPECTED_RACES = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24, 2025: 24}
CACHE_DIR = RAW.parent / "fastf1_cache"
OUTPUT_FILE = RAW / "f1_performance_data_fastf1.csv"
PROGRESS_FILE = RAW / "fastf1_progress.csv"
DELAY_BETWEEN_SESSIONS = 10
DELAY_BETWEEN_RACES = 5


def load_progress():
    if PROGRESS_FILE.exists():
        df = pd.read_csv(PROGRESS_FILE)
        return df, set(zip(df["Season"], df["RoundNumber"]))
    return pd.DataFrame(), set()


def save_progress(frames):
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Season", "RoundNumber", "Driver", "Team"], keep="last")
    df.to_csv(PROGRESS_FILE, index=False)


def session_results(season, rnd, kind, cols, rename):
    session = fastf1.get_session(season, rnd, kind)
    session.load()
    time.sleep(DELAY_BETWEEN_SESSIONS)
    return session.results[cols].rename(columns=rename)


def collect():
    existing, done = load_progress()
    frames = [existing] if len(existing) else []
    failures = []

    for season in SEASONS:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        new_data = False

        for _, event in schedule[schedule["RoundNumber"] > 0].iterrows():
            rnd, name = event["RoundNumber"], event["EventName"]
            if (season, rnd) in done:
                continue
            print(f"{season} R{rnd:02d} {name}...", end=" ", flush=True)
            try:
                quali = session_results(
                    season, rnd, "Q", ["Abbreviation", "TeamName", "Position"],
                    {"Abbreviation": "Driver", "TeamName": "Team", "Position": "QualifyingPos"})
                race = session_results(
                    season, rnd, "R", ["Abbreviation", "TeamName", "Position", "Points"],
                    {"Abbreviation": "Driver", "TeamName": "Team",
                     "Position": "RaceFinishPos", "Points": "RacePoints"})
            except Exception as e:
                if "RateLimit" in str(e) or "429" in str(e):
                    print("rate limited; progress saved, rerun in ~30 min")
                    if frames:
                        save_progress(frames)
                    return None
                print(f"skipped ({e})")
                failures.append(dict(Season=int(season),RoundNumber=int(rnd),Error=str(e)))
                (RAW / "fastf1_failed_events.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
                continue

            merged = quali.merge(race, on=["Driver", "Team"], how="outer")
            merged["Season"], merged["RoundNumber"], merged["RaceName"] = season, rnd, name
            frames.append(merged)
            new_data = True
            print("ok")
            time.sleep(DELAY_BETWEEN_RACES)

        if new_data:
            save_progress(frames)

    if not frames:
        raise RuntimeError("No events downloaded; inspect fastf1_failed_events.json")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Season", "RoundNumber", "Driver", "Team"], keep="last")
    df["QualifyingPos"] = pd.to_numeric(df["QualifyingPos"], errors="coerce")
    df["RaceFinishPos"] = pd.to_numeric(df["RaceFinishPos"], errors="coerce")
    # Grand Prix points only; sprint sessions are not loaded.
    df["RacePoints"] = pd.to_numeric(df["RacePoints"], errors="coerce").fillna(0)
    if df.groupby("Season").RoundNumber.nunique().to_dict() != EXPECTED_RACES:
        save_progress(frames)
        raise RuntimeError("Incomplete event coverage; progress preserved. Rerun to collect failed events.")
    if df.duplicated(["Season","RoundNumber","Driver"]).any():
        raise RuntimeError("Duplicate driver-event identities require review; progress preserved")
    counts = df.groupby(["Season","RoundNumber"]).size()
    if not counts.between(18,22).all():
        raise RuntimeError("Unexpected event row counts; progress preserved for review")
    (RAW / "fastf1_failed_events.json").write_text("[]\n",encoding="utf-8")
    cols = ["Season", "RoundNumber", "RaceName", "Team", "Driver",
            "QualifyingPos", "RaceFinishPos", "RacePoints"]
    return df[cols].sort_values(["Season", "RoundNumber", "QualifyingPos"], ignore_index=True)


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    data = collect()
    if data is not None:
        data.to_csv(OUTPUT_FILE, index=False)
        PROGRESS_FILE.unlink(missing_ok=True)
        print(f"{len(data):,} driver-race rows -> {OUTPUT_FILE}")
