"""Build the sponsor-season panel used in all models.

Output: data/processed/sponsor_panel.csv
One row per sponsor-team-season, 2018-2024. Retained = 1 if the sponsor
appears with the same team the following season.
"""

import pandas as pd

from common import PROCESSED, RAW, build_panel, load_sponsors, corrected_sponsor_records

if __name__ == "__main__":
    sp = load_sponsors()
    panel = build_panel(sp)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / "sponsor_panel.csv", index=False)
    corrected_sponsor_records().to_csv(PROCESSED / "sponsor_records_corrected.csv", index=False)
    sp.to_csv(PROCESSED / "sponsors_matched.csv", index=False)

    raw = pd.read_csv(RAW / "f1_sponsors_standardized.csv")
    print(f"Sponsor records (raw):    {len(raw):,}")
    print(f"After identity matching:  {len(sp):,}")
    print(f"Unique sponsors:          {sp['Key'].nunique():,}")
    print(f"Panel rows (2018-2024):   {len(panel):,}")
    print(f"Continuation rate:        {panel['Retained'].mean():.3f}")
    print(f"Left-censored rows:       {panel['LeftCensored'].sum():,}")
