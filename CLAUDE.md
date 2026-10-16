# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run tests: `pytest`
- Run Streamlit dashboard: `streamlit run app.py` (when created)
- Install dependencies: `pip install -r requirements.txt`

## Architecture
- `src/`: Core Python modules for signal generation, preprocessing, STFT, CNN autoencoder, anomaly scoring, evaluation, and visualization.
- `app.py`: Streamlit interactive scientific dashboard.
- `scripts/`: Execution scripts for demo data generation, model training, and evaluation.
- `tests/`: Automated unit and integration tests using `pytest`.
