# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run tests: `pytest`
- Run Streamlit dashboard: `streamlit run app.py`
- Install dependencies: `pip install -r requirements.txt`

## Architecture
- `src/`: Core Python modules for signal generation, preprocessing, STFT, CNN autoencoder, anomaly scoring, evaluation, and visualization.
- `app.py`: Streamlit entry point for the SETI-AI Observatory dashboard; pages and widgets live in `ui/`, the dashboard data layer in `src/observatory/` (no Streamlit imports there).
- `scripts/`: Execution scripts for demo data generation, model training, and evaluation.
- `tests/`: Automated unit and integration tests using `pytest`.
