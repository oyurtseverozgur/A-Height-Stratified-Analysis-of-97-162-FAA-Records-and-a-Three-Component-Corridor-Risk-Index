# UAM Wildlife Strike Analysis (JATE)

Code and data for the manuscript **"Wildlife Strike Hazard in the Low-Altitude Urban Air Mobility Envelope: A Selection-Corrected Analysis of 97,162 FAA Records and an Exposure-Normalized Corridor Risk Index"** (Journal of Aviation Technology and Engineering, revised submission R1, November 2026).

## Revised analysis (R1) – current
- `uam_wildlife_r1.py` – complete analysis: representativeness tests, selection-bias correction, adjusted logistic model, rotorcraft analysis, seasonal and trend analysis, UAM Corridor Risk Index (UCRI), hold-out validation, sensitivity analyses, Figures 1–7.
- `Supplementary_Material_JATE_R1.zip` – the script plus `data/atads_ops_2010_2025.csv` (FAA ATADS operations, 45 airports) and `results/` (all statistics and tables reported in the manuscript). See the README inside the archive.

To reproduce, download the NWSD public database (Public.xlsx) from https://wildlife.faa.gov, then run:
`python uam_wildlife_r1.py --nwsd Public.xlsx --ops data/atads_ops_2010_2025.csv --out results`
(requires pandas, numpy, scipy, statsmodels, matplotlib, openpyxl).

## Original submission – superseded
`uam_wildlife_strike_analysis.py` and `readme.md.txt` belong to the original submission and are kept unchanged for comparison. Their results are superseded by the R1 analysis.
