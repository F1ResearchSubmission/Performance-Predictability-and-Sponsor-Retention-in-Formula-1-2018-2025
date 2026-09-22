# Performance Predictability and Sponsor Continuation in Formula 1, 2018–2025

Data and code for a study of sponsor continuation across all ten Formula 1 constructors. It asks whether qualifying-to-race predictability is associated with a sponsor staying with a team the next season, beyond conventional performance, and describes how the industry mix of F1 sponsorship changed from 2018 to 2025.

The sponsor dataset includes 2,083 sponsor-team-season records and 643 sponsors.

## Results

Logistic model, n = 1,769 sponsor-seasons (2018–2024 origins). Standard errors clustered by team; p-values use a t-distribution with 9 degrees of freedom.

| | Estimate |
|---|---|
| Continuation rate | 72.1% |
| Predictability, per SD | OR 0.98, 95% CI 0.86–1.13, p = 0.77 |
| Predictability, alternative measure, per SD | OR 1.07, 95% CI 0.84–1.36, p = 0.53 |
| Mean race finish, per position | OR 0.954, 95% CI 0.932–0.976, p = 0.001 |
| Mean race finish, with team indicators | OR 0.933, p = 0.18 |
| Automotive vs Technology | OR 1.83, p = 0.034 |
| Crypto & Web3 vs Technology | OR 0.47, p = 0.017 |
| Held-out AUC, teams held out | 0.573 |

Predictability shows no significant association in any specification. The performance association holds with season indicators but is not significant with team indicators, so it is sensitive to adjustment for persistent team differences. A wild cluster bootstrap of the corresponding linear probability models gives the same conclusions (`output/few_cluster_inference.txt`).

## Layout

```
data/
  raw/                        source files, unmodified (checksums in raw_sha256.csv)
  sponsor_record_corrections.csv   six sourced fixes to individual records
  correction_sources.csv      primary sources for those fixes
  sponsor_crosswalk.csv       name variants -> single sponsor identity
  taxonomy_map.csv            raw industry label -> analysis category
  quarantine/                 2025 performance rows, excluded from analysis
  processed/                  generated
  DATA_DICTIONARY.md
src/
  run_all.py                  runs 02-08
  01_collect_performance.py   optional FastF1 download (network, slow)
  02_aggregate_performance.py team-race and team-season performance, 2018-2024
  03_build_sponsor_panel.py   sponsor-season panel
  04_model.py                 main model, prediction
  05_robustness.py            sponsor-matching and record-correction sensitivity
  06_figure1.py
  07_few_cluster_inference.py t(9) p-values, wild cluster bootstrap
  08_validate_and_sensitivity.py  data checks, team/year specifications
output/                       tables, figure, validation report
```

## Reproducing

Python 3.12.

```
pip install -r requirements.txt
python src/run_all.py
```

Regenerates everything in `data/processed/` and `output/`. The bootstrap takes a few minutes. `output/validation.txt` lists the data checks; all should read PASS.

## Data

**Performance.** Qualifying and race classifications from the official F1 timing feed via FastF1. Team values average both drivers. Only 2018–2024 performance is used: 2025 is never a starting season for a continuation outcome, and some manually entered 2025 results were found to be wrong, so those rows are set aside in `data/quarantine/`.

**Sponsors.** Compiled by hand from team websites, news articles, and livery images. A sponsor was recorded if it appeared on the livery, in team branding, or on an official partner list that season. Continuation = 1 if the same sponsor appears with the same team the following season.

**Identity matching.** Names are matched after removing capitalization, spacing, and punctuation differences, then 38 spelling and naming variants are merged (`sponsor_crosswalk.csv`). Six individual records are corrected against primary sources (`sponsor_record_corrections.csv`), including McLaren's 2019 "Volvo" listing (Volvo Trucks) and Haas's 2020 BlueDEF entry, which had been split into two sponsors.

## Limitations

- Sponsor listings have no per-row source links, and historical completeness is not verified. Changes in recorded composition may partly reflect source coverage. 11.2% of recorded exits reappear with the same team two seasons later; some of these may be listing gaps rather than real departures.
- Some sponsor identities are judgment calls, such as sub-brands versus parent companies. Brand-level merges are tested in `05_robustness.py`.
- Tenure is observed only from 2018 and is a lower bound for sponsors already present that year.
- Ten team clusters limit inference. The performance association is not significant with team indicators. The Automotive estimate sits near p = 0.05: p = 0.060 with every sourced correction except McLaren's 2019 Volvo Trucks listing, and 0.034 with all six (`output/robustness.txt`).
- Continuation is presence in consecutive seasons, not a confirmed contract renewal.
- Points are Grand Prix points; sprint points are not included.
- All results are associations from observational data.

## License

Code: MIT. Data: CC BY 4.0.
