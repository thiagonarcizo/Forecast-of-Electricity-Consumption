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
    col1, col2 = st.columns(2)

    model_st_tab1, model_st_tab2 = st.tabs(["KNearestNeighbors Model", "RandomForest Model"])
    with model_st_tab1:
        st.markdown("""
### 1. Data Split by ACORN
- **Sort & index**  
  - Sort half‐hourly data by `Acorn` and `DateTime`.  
  - Assign a within‐group index (`group_idx`) for each ACORN.

- **48‐hour test horizon**  
  - For each ACORN, let _N₁_ = total rows in that group.  
  - Mark the final 96 rows (`group_idx ≥ Nᵢ − 96`) as test (48 hours), and the rest as train.

### 2. Preprocessing & Pipeline (per ACORN)
- **Features**  
  - Numeric:  
    ```
    visibility, windBearing, temperature, dewPoint, pressure, windSpeed, humidity,
    nb_clients, hour, dayofweek, month, dayofyear, is_holiday
    ```  
  - Categorical:  
    ```
    precipType, icon
    ```

- **Pipeline Steps**  
  1. **StandardScaler** on all numeric features.  
  2. **OneHotEncoder** on each categorical feature.  
  3. **KNeighborsRegressor** (hyperparameters tuned per ACORN).

### 3. Hyperparameter Tuning with Optuna (per ACORN)
- **Search Space**  
  - `algorithm`: auto, ball_tree, kd_tree, brute  
  - `metric`: euclidean, manhattan, chebyshev, minkowski  
    - If metric = minkowski, then _p_ ∈ {1, 2}  
  - `n_neighbors`: integer from 1 to 100  
  - `weights`: uniform, distance  
  - `leaf_size`: integer from 10 to 60

- **Tuning Process**  
  1. For each trial, instantiate a KNN regressor with the sampled hyperparameters.  
  2. Perform 3‐fold time‐series cross‐validation on the ACORN’s training subset.  
  3. Optimize average RMSE (root mean squared error) across folds.

- **Outcome**  
  - **Best Hyperparameters** (per ACORN)  
  - **Cross‐Validated RMSE**

### 4. Final Training & Evaluation (per ACORN)
1. **Retrain** on all train rows with the selected hyperparameters.  
2. **Predict** consumption over the final 48 hours (test set).  
3. **Compute Metrics**  
   - **RMSE** 
   - **MAE**
   - **MAPE**

### 5. Feature Importance (Permutation, per ACORN)
- **Procedure**  
  1. Permute each feature in the test subset and measure increase in RMSE.
  2. Remove columns with negative importance because they might harm the result (`visibility` feature)
                  
### Results for test set
  """   )
        metrics_knn_df = pd.DataFrame({
                'ACORN': ['C', 'P', 'F'],
                'MAE': [0.0410, 0.0337, 0.0370],
                'MAPE (%)': [30.8348, 30.8438, 30.8438], 
                'RMSE': [0.0512, 0.0596, 0.0448]
            })
        
        st.dataframe(metrics_knn_df)

        st.image("src/img/plot1knn.png")

        st.markdown("After such bad results, this model was discarded as an option")
    
    
    with model_st_tab2:
        st.markdown(
            """
### 1. Data Split by ACORN
- **Sort & index**  
  - Sort half‐hourly data by `Acorn` and `DateTime`.  
  - Assign a within‐group index (`group_idx`) for each ACORN.

- **48‐hour test horizon**  
  - For each ACORN, let _Nᵢ_ = total rows in that group.  
  - Mark the final 96 rows (`group_idx ≥ Nᵢ − 96`) as test (48 hours), and the rest as train.

### 2. Preprocessing & Pipeline (per ACORN)
- **Features**  
  - Numeric:  
    ```
    visibility, windBearing, temperature, dewPoint, pressure, windSpeed, humidity,
    nb_clients, hour, dayofweek, month, dayofyear, is_holiday
    ```  
  - Categorical:  
    ```
    precipType, icon
    ```

- **Pipeline Steps**  
  1. **StandardScaler** on all numeric features.  
  2. **OneHotEncoder** on each categorical feature.  
  3. **RandomForestRegressor** (hyperparameters tuned per ACORN).

### 3. Hyperparameter Tuning with Optuna (per ACORN)
- **Search Space**  
  - `n_estimators`: integer from 50 to 300  
  - `max_depth`: integer from 5 to 30  
  - `min_samples_split`: integer from 2 to 20  
  - `min_samples_leaf`: integer from 1 to 10  
  - `max_features`: choice of {“sqrt”, “log2”, None}

- **Tuning Process**  
  1. For each trial, instantiate a Random Forest with the sampled hyperparameters.  
  2. Perform 3‐fold time‐series cross‐validation on the ACORN’s training subset.  
  3. Optimize average RMSE (root mean squared error) across folds.

- **Outcome**  
  - **Best Hyperparameters** (per ACORN)  
  - **Cross‐Validated RMSE**

### 4. Final Training & Evaluation (per ACORN)
1. **Retrain** on all train rows with the selected hyperparameters.  
2. **Predict** consumption over the final 48 hours (test set).  
3. **Compute Metrics**  
   - **RMSE** 
   - **MAE**:
   - **MAPE**:

### Results for the test set
"""
        )

        metrics_rf_df = pd.DataFrame({
                'ACORN': ['C', 'P', 'F'],
                'MAE': [0.0185, 0.0151, 0.0127],
                'MAPE (%)': [1.8498, 1.5087, 1.2724], 
                'RMSE': [0.0274, 0.0201, 0.0164]
        })
        st.dataframe(metrics_rf_df)
        st.image("src/img/plotrf1.png")
        st.image("src/img/plotrf2.png")
        st.image("src/img/plotrf3.png")
        st.markdown(
            """
    After discussion with one of the professors, we realized that a 48h window might be too short to calculae errors.
    So we retrained the model and increased the test set to a week. The results are below and indicate that the model
    is not as good as the previous results indicated.
"""
        )
        metrics_rf_df = pd.DataFrame({
                'ACORN': ['C', 'P', 'F'],
                'MAE': [0.0192, 0.0269, 0.0117],
                'MAPE (%)': [7.1835, 18.1243, 5.7277], 
                'RMSE': [0.0257, 0.0386, 0.0145]
        })
        st.dataframe(metrics_rf_df)
        st.markdown(
            """
    ### Prediction for 13/01/2014 and 14/01/2014
"""
        )

        st.image("src/img/plotrf4.png")
        st.image("src/img/plotrf5.png")
        st.image("src/img/plotrf6.png")


with med_term_tab:
    st.header("Medium Term Forecast")
    col1, col2 = st.columns(2)

    model_tab1, model_tab2, model_tab3, model_tab4, model_tab5 = st.tabs(["LightGBM Model", "LSTM Model", "SARIMAX Model", "MLP Model", "SVM Model"])
    
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

        st.markdown("""
## Static Forecast

### Building the Future Dataset
- Create one row per day between the start and end dates.
- Extract simple date features (weekday, month, day of year).
- Encode cyclical patterns (weekday, month, day of year) using sine/cosine.
- Add flags for weekend, month start, and month end.
- Estimate winter weather metrics by drawing around historical winter averages.
- Compute combined weather measures when possible (e.g., temperature x humidity, temperature range).
- Fix client count to its historical average if available.
- Initialize a placeholder consumption value from the recent historical average.
""")

        st.markdown("""
## Rolling Forecast

### Computing Lag and Rolling Metrics
- Maintain a chronological list of past consumption values (historical + any prior predictions).
- For lagged values:
  - Extract the consumption from 1, 2, 3, 7, 14, 21, and 28 days ago (or use the earliest available if not enough history).
- For rolling windows:
  - For window lengths of 3, 7, 14, 21, and 30 days:
    - Calculate mean, standard deviation, minimum, and maximum over the most recent values (or all available data if the window exceeds history length).
- For exponential moving averages:
  - Compute an exponential average with smoothing factors (e.g., 0.1, 0.3, 0.5) over the entire historical series.

### Iterative Day-by-Day Prediction
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
""")

        st.markdown("""
## Comparison

- **Static Forecast**
  - Uses the same set of historical features for every day in the horizon.
  - Lag and trend inputs remain constant, ignoring how actual or predicted usage evolves.
  - May drift if underlying patterns shift mid-horizon.
- **Rolling Forecast**
  - Updates lag, rolling, and exponential-averaged features each day using prior predictions.
  - Reflects the latest trends, adapting to sudden changes or emerging patterns.
  - Mimics real-world deployment—only past values (including new forecasts) inform each next prediction.

**Because the rolling approach continually incorporates the most recent information and adjusts forecasts dynamically, it typically yields more accurate and realistic results than a static projection.**
""")
        
        st.image("src/img/plot2lgb.png")

        st.header("Future Predictions - Rolling Forecast")

        st.image("src/img/plot5lgb.png")
        st.image("src/img/plot4lgb.png")

    with model_tab2:
        st.subheader("LSTM Forecast") 
        st.markdown("30-day forecast using LSTM model")

    with model_tab3:
        st.subheader("SARIMAX")
        st.markdown("30-day forecast using Prophet model")

    with model_tab4:
        st.subheader("MLP")
        st.markdown("30-day forecast using MLP model")
        st.markdown(
            """
## Medium‐Term Load Forecasting with MLP (One Unified Model)

### 1. Data and Feature Engineering
- **Data Sources**  
  - Daily consumption (`Conso_kWh`) for days \([0, t]\).  
  - Daily weather features (temperature, humidity, wind speed, precipitation, etc.) for days \([0, t+30]\).  
  - ACORN group code (categorical) per daily record.  
  - Calendar features derived from each date:  
    - Day of week, day of year, month, weekend/weekday flag, holiday flag.

- **Construct Combined DataFrame**  
  - Merge consumption and historical weather for days ≤ t.  
  - Append 30‐day “future” rows (dates \(t+1\) to \(t+30\)) with weather forecasts and empty consumption.

- **Final Feature Set**  
  - **Numeric**:  
    ```
    temperature, humidity, windSpeed, precipitation,
    dayofweek, dayofyear, month, is_weekend, is_holiday, nb_clients
    ```  
  - **Categorical**:  
    ```
    Acorn, windBearing, icon, precipType
    ```  
  - Drop any rows with missing weather or consumption (for training).

### 2. Train/Test Split
- **Define Training Window**  
  - For each ACORN, let \(t\) = last date with known consumption.  
  - **Training Set**: All rows where `Date ≤ t`.  
  - **Forecast Set**: Rows where \(t < \text{Date} ≤ t+30\) (weather known, consumption unknown).

### 3. Preprocessing Pipeline
- **Scaling & Encoding**  
  - **StandardScaler** on all numeric features (zero mean, unit variance).  
  - **OneHotEncoder** on each categorical feature (`handle_unknown="ignore"`).

- **Pipeline Structure**  
  1. Apply scaling and one‐hot encoding.  
  2. Forward transformed features into an `MLPRegressor`.

### 4. MLP Model Configuration
- **Initial Architecture**  
  - Two hidden layers (e.g.\ 100 → 50 neurons).  
  - Activation: ReLU; Solver: Adam with adaptive learning rate.  
  - Weight‐decay (alpha) set to a small default.

- **Hyperparameters to Tune**  
  - **Hidden layers**: number of layers (1–3), neurons per layer (32–256).  
  - **Regularization (alpha)**: \(10^{-6}\) – \(10^{-2}\).  
  - **Learning rate (`learning_rate_init`)**: \(10^{-5}\) – \(10^{-2}\).  
  - **Learning rate schedule**: constant vs. adaptive.  
  - **Activation**: ReLU vs. tanh.  
  - **Solver**: Adam vs. L-BFGS.  
  - **Batch size**: 32, 64, 128.  
  - **Tolerance (`tol`)**: \(10^{-5}\) – \(10^{-3}\).

- **Optimization**  
  - Use Optuna (50 trials) with TimeSeriesSplit (3 folds) on days \(≤ t\).  
  - Objective: Minimize cross‐validated RMSE.

### 5. Final Training and Evaluation
1. **Retrain** on all days ≤ t using best hyperparameters.  
2. **Predict** consumption for days \(t+1\) to \(t+30\).  
3. **Compute Metrics on Hold‐Out (last 30 historical days)**:  
   - **RMSE**, **MAE**, **MAPE**.  
4. **Forecast Submission**: For days \(t+1\) – \(t+30\), report predicted values.

### 6. Feature Importance (Permutation)
- **Procedure**  
  1. On hold‐out days \([t-29, t]\), permute each feature column and measure increase in RMSE.  
  2. Repeat multiple permutations to obtain mean importance + standard deviation.

- **Visualization Placeholder**  

"""
        )

    with model_tab5:
        st.subheader("SVM")