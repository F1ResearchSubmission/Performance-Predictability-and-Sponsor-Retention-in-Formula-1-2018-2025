# Data Dictionary

## Constructors

Ten continuous entities. Rebranded teams are treated as one team.

| Entity | Names in source data |
|---|---|
| Red Bull Racing | Red Bull Racing |
| Mercedes | Mercedes |
| Ferrari | Ferrari |
| McLaren | McLaren |
| Alpine/Renault | Renault, Alpine |
| Aston Martin/Racing Point/Force India | Force India, Racing Point, Aston Martin |
| RB/AlphaTauri/Toro Rosso | Toro Rosso, AlphaTauri, RB, Racing Bulls |
| Sauber/Alfa Romeo | Sauber, Alfa Romeo Racing, Alfa Romeo, Kick Sauber |
| Haas | Haas F1 Team |
| Williams | Williams |

---

## raw/

Unmodified source files. SHA-256 checksums in `raw_sha256.csv`; `src/08` verifies them on every run.

**f1_sponsors_original.csv** — 2,082 sponsor-team-season listings, compiled by hand. Columns: `Team`, `Season`, `SponsorName`, `SponsorIndustry`.

**f1_sponsors_standardized.csv** — the same listings with team names mapped to the entities above. Sponsor names and industry labels are unchanged. This is the file the pipeline reads.

**f1_performance_data_2018_2025_complete.csv** — 3,459 driver-race rows across 173 races. Columns: `Season`, `RoundNumber`, `RaceName`, `Team`, `Driver`, `QualifyingPos`, `RaceFinishPos`, `RacePoints` (Grand Prix points; sprints excluded), `TeamStd` (constructor entity). 2018 through 2025 round 16 were downloaded via FastF1. 2025 rounds 17–24 were entered manually and contain errors; see `quarantine/`. A few positions are missing, and team means use whichever drivers have a value.

## Corrections

**sponsor_record_corrections.csv** — six fixes to individual records, applied before name matching. Each is limited to one team, season, and name. Columns: `ID`, `Action` (rename or add), `Team`, `Season`, `OldName`, `NewName`, `Industry`, `SourceID`, `Reason`.

**correction_sources.csv** — the primary source behind each correction. Columns: `SourceID`, `Publisher`, `PublicationDate`, `CheckedDate`, `URL`, `Supports`. Covers these corrections only, not every listing.

**sponsor_crosswalk.csv** — name variants merged into one identity after normalizing case, spacing, and punctuation. `typo`: spelling or naming variants of the same sponsor, used in all results (38). `brand`: sub-brands and product lines, used only in the robustness check (11). `keep`: similar names that are different companies, never merged (5). A scoped record correction can override a `keep`.

**taxonomy_map.csv** — 61 raw industry labels mapped to 17 analysis categories: Apparel & Sportswear, Automotive, Business Services, Consumer Goods & Retail, Crypto & Web3, Cybersecurity, Energy, Financial Services, Food & Beverage, Industrial & Aerospace, Luxury & Fashion, Luxury Watches, Media & Entertainment, Technology, Telecom, Travel, Hospitality & Logistics, and Other (categories too small to model).

## quarantine/

**performance_2025_not_used.csv** — all 2025 performance rows. Some manually entered results contradict the official classifications, so none of these rows is used. 2025 sponsor listings are still used to determine whether 2024 sponsors continued and for 2025 composition.

---

## processed/ (generated)

**sponsor_records_corrected.csv** — 2,083 records after corrections, with `OriginalSponsorName` and `CorrectionID`.

**sponsors_matched.csv** — corrected records with `Key` (sponsor identity) and `Industry`. Each sponsor gets its most frequent category across all seasons; ties go to the earliest. Duplicate listings of one sponsor in the same team-season are collapsed.

**sponsor_panel.csv** — one sponsor with one team in a season, 2018–2024. 1,769 rows. Input to all models.

| Column | Description |
|---|---|
| Team, Season, Key | Sponsor-team-season identifier |
| SponsorName | Recorded name |
| Industry | Analysis category |
| Retained | 1 if the sponsor is listed with the same team the next season |
| Tenure | Consecutive seasons with the team up to and including this one |
| LeftCensored | 1 if the run reaches back to 2018, so true tenure is unknown |
| LogTenure | Natural log of Tenure |
| TeamSponsorCount | Sponsors listed for the team that season |
| MeanRaceFinishPos, MeanQualifyingPos | Season means of team-average positions |
| QualRaceCorrelation | Within-season Pearson r between team-average qualifying and race position |
| PositionChange_Std | SD of qualifying minus finishing position |
| PredStability | Negated PositionChange_Std; higher = more consistent position changes |
| PointsPerRace | Grand Prix points per race |
| SeasonCentered | Season minus 2021 |

**f1_team_race_level_2018_2024.csv** — 1,490 team-race rows: average qualifying and race position, summed points, drivers with data.

**f1_team_season_metrics_2018_2024.csv** — 70 team-seasons: position means and SDs, `QualRaceCorrelation` and its p-value, position-change mean and SD, points. Each correlation comes from 17–24 races; that estimation uncertainty is not carried into the sponsor model.

---

## output/ (generated)

| File | Contents |
|---|---|
| results.txt, model_coefficients.csv | Main logistic model: odds ratios, t(9) confidence intervals and p-values, per unit and per SD |
| few_cluster_inference.txt, bootstrap_results.csv | Wild cluster bootstrap of linear probability models (coefficients are not odds ratios) |
| specification_sensitivity.csv | Main model with season indicators, team indicators, and both |
| robustness.txt | Results under four sponsor-matching rules and under subsets of the record corrections; reappearance rate |
| prediction_folds.csv | AUC for each of five team-grouped folds; the reported AUC is their mean |
| composition_counts.csv, composition_percent.csv | Listings per category per season, and shares |
| figure1_composition.png, .svg | Figure 1 |
| summary.json | Headline counts and metrics |
| validation.txt | Data and pipeline checks |
| environment.json | Package versions used |

A sponsor that is absent one season and listed again the next may be a returning partner or a gap in the listings. The reappearance rate in `robustness.txt` is not an estimate of listing errors.
