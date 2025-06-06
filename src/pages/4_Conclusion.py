import streamlit as st

st.set_page_config(
    page_title="Conclusion",
    page_icon="📜",
)

st.title("📜 Conclusion")

st.markdown(
    """
The exploratory analysis highlighted clear consumption patterns across the ACORN
segments. Weekends showed higher average usage, particularly for ACORN‑F and
ACORN‑C households, and winter months consistently drove the largest loads.
Holidays produced noticeable spikes for higher‑consumption groups, while lower
consumption households remained comparatively stable. These findings guided our
feature engineering and model selection for the forecasting stages.

For short‑term forecasting we evaluated a K‑Nearest Neighbors approach but the
results were unsatisfactory, with MAPE values above 30%. We therefore turned to
a Random Forest model. After hyperparameter tuning and separate training per
ACORN group, the Random Forest delivered much better accuracy on the initial
48‑hour test window, with MAPE around 1–2%. Even when the evaluation horizon was
extended to one week, performance remained acceptable. This robustness, coupled
with the model's ability to handle numerous weather and calendar predictors,
made the Random Forest our preferred choice for predicting half‑hourly demand.

Medium‑term forecasting involved comparing several algorithms: LightGBM, LSTM,
SARIMAX, MLP and SVM. Static versions of these models performed reasonably well,
with LightGBM achieving low error rates, yet the LSTM architecture offered a
more natural way to capture temporal dependencies. We implemented a rolling
training strategy where the LSTM is updated day by day using previous
predictions. This feedback loop allowed the model to adapt to evolving trends
and produced MAPE values below 4.5% across the ACORN segments—better than the
static LSTM and the other candidates.

Taking these results together, we retained the Random Forest model for short
term (48‑hour to one‑week) predictions and adopted the LSTM with rolling
forecasting for the monthly horizon. The Random Forest excels at short lead
times thanks to its ensemble of decision trees, whereas the rolling LSTM offers a flexible, sequential approach that captures medium‑term dynamics more faithfully.
"""
)
