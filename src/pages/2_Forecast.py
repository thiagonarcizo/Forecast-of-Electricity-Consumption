import streamlit as st

st.set_page_config(
    page_title="Forecast",
    page_icon="🔮",
)

st.title("🔮 Forecast")

st.markdown("This page contains the forecast of electricity consumption.")

short_term_tab, long_term_tab = st.tabs(["Short Term", "Long Term"])

with short_term_tab:
    st.header("Short Term Forecast")
    st.markdown("This section will be populated with the short term forecast, based on the `03_modelisation_court_terme_b.ipynb`, `model_daily.ipynb` and `model2_daily.ipynb` notebooks.")
    st.markdown("Please provide the relevant code snippets or a summary of the models from the notebooks to proceed.")
    # To be filled with Short Term Forecast content

with long_term_tab:
    st.header("Long Term Forecast")
    st.markdown("This section will be populated with the long term forecast, based on the `03_long_term_forecasting.ipynb` notebook.")
    st.markdown("Please provide the relevant code snippets or a summary of the steps from the notebook to proceed.")
    # To be filled with Long Term Forecast content 