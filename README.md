# Forecast of Electricity Consumption

This repository contains the coursework for the **EI-ST4** project. The objective is to forecast UK household electricity usage with models tailored to different customer segments (ACORN categories). Results are showcased through an interactive Streamlit application.

## Project Phases
1. **Analysis and characterisation** – Explore consumption data, quantify weather effects and identify temporal patterns such as seasonality and autocorrelation.
2. **Short-term forecasting** – Predict half‑hourly usage for the next 48 hours using historical consumption, weather and calendar features.
3. **Medium-term forecasting** – Forecast aggregated daily consumption over one month. Training uses data preceding the prediction window and evaluation mirrors the short-term approach.
4. **Interactive dashboard** – Visualise exploratory findings and forecasts in an intuitive Streamlit interface.

## Data
- Pre‑processed half‑hourly and daily load profiles split by ACORN segment
- Weather observations (temperature, humidity, wind…) matched to the consumption time series
- UK public holiday calendar

The raw and processed data sets reside in the `data/` directory.

## Evaluation Criteria
- Quality of the exploratory analysis and feature engineering
- Modelling rigour and justification of choices
- Forecast accuracy for both horizons
- Clarity of the results presentation and the interactive dashboard
- Team collaboration and project management

We adopt a simple‑to‑complex modelling strategy, iterating often and validating each step carefully.

## Running the Application
Install the dependencies listed in `requirements.txt` and start the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run src/Introduction.py
```

## Repository Structure
- `src/` – Streamlit pages and helper modules
- `data/` – raw, interim and processed data sets
- `notebooks_*` – exploratory work from team members
- `sujet/` – project description document

Feel free to explore the notebooks and contribute via pull requests.

