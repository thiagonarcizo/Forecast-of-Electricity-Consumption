import streamlit as st

st.set_page_config(
    page_title="Forecast of Electricity Consumption",
    page_icon="⚡",
)

st.title("⚡ Forecast of Electricity Consumption")

st.markdown(
    """
    This Streamlit application presents our project on forecasting
    electricity consumption for UK households. The work focuses on the
    **ACORN** customer segments and makes use of pre‑processed load
    profiles together with weather data and information about public
    holidays.

    ### Project Phases
    1. **Analysis and characterisation** – Explore the data, quantify the
       impact of meteorological conditions (especially temperature) and
       identify key temporal patterns such as autocorrelation and
       seasonality.
    2. **Short-term forecasting** – Build a model that predicts
       half-hourly consumption for the next 48 hours. Historical data
       prior to the forecast window are used for training and validation
       and the model performance is assessed rigorously.
    3. **Medium-term forecasting** – Develop a model to forecast the
       aggregated daily consumption over a one-month horizon. Training
       relies on historical data before the target month and the results
       are evaluated in the same manner as the short-term model.
    4. **Interactive dashboard (bonus)** – Provide a clear and useful
       visualisation of the forecasts through a Streamlit dashboard.

    ### Data
    * Pre-processed electricity load profiles by ACORN segment.
    * Historical weather observations (temperature, humidity, wind
      speed…) aligned with the consumption time series.
    * Calendar information about UK public holidays.

    ### Evaluation criteria
    * Quality of the exploratory analysis.
    * Rigour and justification of the modelling approach and feature
      engineering.
    * Forecast accuracy for both short- and medium-term horizons.
    * Clarity and relevance of the results presentation – including the
      optional interactive dashboard.
    * Collaboration and project management aspects.

    We follow the recommended approach of starting with simple baseline
    models, iterating and communicating results regularly, and
    critically assessing each modelling choice.
    """
)
