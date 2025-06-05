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

        st.image("img/plot1lgb.png")

    with model_tab2:
        st.subheader("LSTM Forecast") 
        st.markdown("30-day forecast using LSTM model")