# BBO Capstone Project

## Overview
This project tackles a black-box optimisation challenge with 8 unknown functions.
Goal: maximise each function using sequential queries (one per function per week).

## Repository Structure
- `data/` – initial data and weekly results
- `notebooks/` – query generation per round
- `src/` – reusable optimisation code
- `reports/` – final results, plots, and reflection
- `docs/` – datasheet and model card

## How to Reproduce
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks in order: `01_data_exploration.ipynb` → `02_round1...`
3. All results are saved in `data/results/`

## Results
- Best outputs are summarised in `reports/final_results_table.png`
- Progress plots in `reports/figures/`

## Documentation
- [Datasheet](datasheet.md)
- [Model Card](model-card.md)
