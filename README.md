# Forecast of Electricity Consumption

This repository contains the coursework for the **EI-ST4** project. The objective is to forecast UK household electricity usage with models tailored to different customer segments (ACORN categories). Results are showcased through an interactive Streamlit application composed of several pages.

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

## Project Architecture
The main folders at the repository root are:

- **src/** – Streamlit application and supporting modules
- **data/** – raw, interim and processed data sets
- **models/** – saved machine learning artefacts
- **notebooks/** – **DEFINITIVE** notebooks
- **notebooks_alexsandro/** – experiments by Alexsandro
- **notebooks_gustavo/** – experiments by Gustavo
- **notebooks_martin/** – experiments by Martin
- **notebooks_thiago/** – experiments by Thiago
- **sujet/** – project description document
- `requirements.txt` and `setup.py` – dependency management

Within **src/pages** reside the Streamlit pages:
`1_EDA.py`, `2_Forecast.py`, `3_Dashboard.py` and `4_Conclusion.py`.

## Streamlit Pages
The [web interface](https://group-4.streamlit.app/) is organised into the following sections:

1. **Introduction** – project outline and objectives.
2. **EDA** – exploratory analysis with interactive visualisations of the ACORN groups.
3. **Forecast** – short- and medium-term prediction results with explanations of each modelling approach.
4. **Dashboard** – compare historical and predicted daily consumption for any model.
5. **Conclusion** – summary of findings and the rationale for the final modelling choices.

The application ultimately selects **Random Forest** for short-term forecasting and a **rolling LSTM** strategy for medium-term forecasts as these models achieved the most reliable accuracy across the ACORN segments.
### Evaluation of Selected Models

**Random Forest (48h test)**

| ACORN | MAE | MAPE (%) | RMSE |
|------|------|---------|------|
| C | 0.0185 | 1.8498 | 0.0274 |
| P | 0.0151 | 1.5087 | 0.0201 |
| F | 0.0127 | 1.2724 | 0.0164 |

**Rolling LSTM (30-day test)**

| ACORN | MAE | MAPE (%) | MSE |
|------|------|---------|------|
| C | 0.6392 | 4.50 | 0.8968 |
| P | 0.2938 | 4.07 | 0.2516 |
| F | 0.4093 | 3.88 | 0.1423 |



