import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Forecast",
    page_icon="🔮",
)

st.title("🔮 Forecast")

st.markdown("This page contains the forecast of electricity consumption.")

short_term_tab, med_term_tab = st.tabs(["Short Term", "Medium Term"])

with short_term_tab:
    st.header("Short Term Forecast")
    st.markdown("This section will be populated with the short term forecast, based on the `03_modelisation_court_terme_b.ipynb`, `model_daily.ipynb` and `model2_daily.ipynb` notebooks.")
    st.markdown("Please provide the relevant code snippets or a summary of the models from the notebooks to proceed.")
    # To be filled with Short Term Forecast content

with med_term_tab:
    st.header("Medium Term Forecast")
    col1, col2 = st.columns(2)

    model_tab1, model_tab2 = st.tabs(["LightGBM Model", "LSTM Model"])
    
    with model_tab1:
        st.subheader("LightGBM Forecast")
        st.markdown("30-day forecast using LightGBM model")
        
        metrics_df = pd.DataFrame({
            'ACORN': ['C', 'P', 'F'],
            'MAE': [0.2227, 0.0997, 0.1152],
            'MAPE (%)': [1.73, 1.49, 1.18], 
            'MSE': [0.1095, 0.0166, 0.0211],
            'R² Score': [0.9444, 0.9758, 0.9790]
        })
    
        st.dataframe(metrics_df)
    
        st.markdown("""
    The LightGBM model was trained with the following approach:
    
    1. **Data Preparation**:
        - Features were prepared excluding Date and target columns
        - Categorical columns were handled appropriately
        - Missing values were filled using forward/backward fill
    
    2. **Model Configuration**:
        - Used LightGBM Regressor with optimized parameters:
        - 2000 estimators (trees)
        - Learning rate of 0.01
        - Max tree depth of 8
        - 31 leaves per tree
        - 80% subsample and column sample rates
        
    3. **Training Process**:
        - Early stopping with 100 rounds patience
        - Validation set used to prevent overfitting
        - Separate models trained for each ACORN group
        
    4. **Results**:
        - Excellent R² scores (>0.94) across all groups
        - Low MAPE (<2%) indicating high accuracy
        - ACORN F showed best performance with 1.18% MAPE
    """)
        st.image("src/img/plot3lgb.png")

        st.header("Static Forecast vs. Rolling Forecast")

        st.markdown('''
    ## Static Forecast

    *Building the Future Dataset*
	- Create one row per day between the start and end dates.
	- Extract simple date features (weekday, month, day of year).
	- Encode cyclical patterns (weekday, month, day of year) using sine/cosine.
	- Add flags for weekend, month start, and month end.
	- Estimate winter weather metrics by drawing around historical winter averages.
	- Compute combined weather measures when possible (e.g., temperature x humidity, temperature range).
	- Fix client count to its historical average if available.
	- Initialize a placeholder consumption value from the recent historical average.
                    
    ## Rolling Forecast
                    
    *Computing Lag and Rolling Metrics*
    - Maintain a chronological list of past consumption values (historical + any prior predictions).
    - For lagged values:
        - Extract the consumption from 1, 2, 3, 7, 14, 21, and 28 days ago (or use the earliest available if not enough history).
    - For rolling windows:
        - For window lengths of 3, 7, 14, 21, and 30 days:
    - Calculate mean, standard deviation, minimum, and maximum over the most recent values (or all available data if the window exceeds history length).
    - For exponential moving averages:
     - Compute an exponential average with smoothing factors (e.g., 0.1, 0.3, 0.5) over the entire historical series.

    *Iterative Day-by-Day Prediction*
    - Announce the start and end dates for the rolling forecast.
    - Initialize the consumption history list from sorted historical data.
    - For each date in the forecast horizon:
    - Build that day's base features (calendar + weather estimates + client count).
     - Compute lag and rolling features using the current consumption history.
    - Merge base and lag/rolling features into a single feature set.
    - Ensure all model inputs are present, filling any missing ones with zero.
    - Run the pre-trained model to predict that day's consumption.
    - Append the new prediction to the consumption history for future lag calculations.
    - Record the date and predicted value for later analysis.
        ''')

        st.image("src/img/plot2lgb.png")

        st.markdown('''
    ## Comparison
    - Static Forecast
        - Uses the same set of historical features for every day in the horizon.
        - Lag and trend inputs remain constant, ignoring how actual or predicted usage evolves.
        - May drift if underlying patterns shift mid-horizon.
	- Rolling Forecast
        - Updates lag, rolling, and exponential-averaged features each day using prior predictions.
        - Reflects the latest trends, adapting to sudden changes or emerging patterns.
        - Mimics real-world deployment—only past values (including new forecasts) inform each next prediction.

**Because the rolling approach continually incorporates the most recent information and adjusts forecasts dynamically, it typically yields more accurate and realistic results than a static projection.**
                    ''')

    with model_tab2:
        st.subheader("LSTM Forecast") 
        st.markdown("30-day forecast using LSTM model")