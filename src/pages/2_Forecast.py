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
        st.subheader("KNearestNeighbors Forecast")
        st.markdown("""
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
        st.subheader("RandomForest Forecast")
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
        st.subheader("LSTM Static Forecast") 
        st.markdown("30-day forecast using static LSTM model")

        lstm_metrics_df = pd.DataFrame({
            'ACORN': ['C', 'P', 'F'],
            'MAE': [0.7567, 0.4291, 0.4209],
            'MAPE (%)': [5.39, 5.93, 4.00],
            'MSE': [0.1041, 0.2782, 0.2563],
        })

        st.dataframe(lstm_metrics_df)

        st.markdown("""
    The LSTM model was trained with the following approach:

    1. **Data Preparation**:
      - Selected numeric features (lag values, weather, calendar flags) as the LSTM inputs.
      - Scaled all inputs to [0,1] using MinMaxScaler trained on the historical window.
      - Created 7-day look-back sequences—each sample uses the previous 7 days of data.

    2. **Model Configuration**:
      - Single-layer LSTM with 64 hidden units.
      - Dropout of 0.2 after the final LSTM output.
      - Dense → ReLU → Dense (32 units hidden, then 1 output).
      - Batch size: 16, learning rate: 0.001, trained for 1000 epochs.

    3. **Training Process**:
      - Trained once on the entire training set (static): no re­training during the test horizon.
      - Early stopping was not used; training ran to completion.
      - Separate LSTM models were fit for each ACORN group (“C”, “P”, “F”).

    4. **Forecast Method**:
      - After training, the model generated all 30 forecasted values in one pass over the test sequences.
      - Test sequences were prepared by sliding a 7-day window across the test period (no new data fed back).
      - Generated a single static forecast vector of length 30 for each ACORN.

    5. **Results**:
      - MAPE values below 6%, showing an average accuracy.
      - ACORN F had the best MAPE at 4.00%.
        """)
        st.image("src/img/plot1lstm.png")
        st.image("src/img/plot2lstm.png")
        st.image("src/img/plot3lstm.png")
        st.image("src/img/plot4lstm_rolling.png")
        st.image("src/img/plot5lstm_rolling.png")
        st.image("src/img/plot6lstm_rolling.png")

        

        st.subheader("LSTM Rolling Forecast")
        st.markdown("30-day forecast using rolling LSTM model")

        rolling_metrics_df = pd.DataFrame({
            'ACORN': ['C', 'P', 'F'],
            'MAE': [0.6392, 0.2938, 0.4093],
            'MAPE (%)': [4.50, 4.07, 3.88],
            'MSE': [0.8968, 0.2516, 0.1423],
        })

        st.dataframe(rolling_metrics_df)

        st.markdown("""
    The Rolling LSTM model was trained with the following approach:

    1. **Data Preparation**:
      - Same numeric features as the static model, scaled with MinMaxScaler.
      - Created 7-day look-back sequences for each sample.

    2. **Model Configuration**:
      - Identical architecture to the static LSTM (64 hidden units, dropout 0.2, Dense→ReLU→Dense).
      - Batch size: 16, learning rate: 0.001, trained for a reduced number of epochs at each rolling step.

    3. **Training Process**:
      - Started with the full training set and generated initial forecasts for the test period.
      - At each test-day, the LSTM was re-trained (for a few epochs) on an expanding window that included prior actuals and any previous predictions.
      - Separate rolling‐trained LSTM models were maintained for each ACORN group.

    4. **Forecast Method**:
      - Predict day 1 of the test horizon using the last 7 days of true history.
      - Append this prediction to the history, drop the oldest day, form a new 7-day window to predict day 2.
      - Repeat iteratively through the 30-day horizon, so each forecast uses the latest available (actual or predicted) data.
      - This feedback loop adapts lag-based inputs dynamically.

    5. **Results**:
      - Rolling MAPE values below 4.5%, demonstrating improved accuracy over the static approach.
      - ACORN F achieved the lowest MAPE at 3.20%.
            """)
        st.image("src/img/plot1lstmrf.png")
        st.image("src/img/plot2lstmrf.png")
        st.image("src/img/plot3lstmrf.png")
        st.image("src/img/plot4lstmrf.png")
        st.image("src/img/plot5lstmrf.png")
        st.image("src/img/plot6lstmrf.png")
        st.image("src/img/plot7lstmrf.png")
        st.image("src/img/plot8lstmrf.png")
        st.image("src/img/plot9lstmrf.png")



    with model_tab3:
        st.subheader("SARIMAX")
        st.markdown("30-day forecast using Prophet model")

        sarimax_metrics_df = pd.DataFrame({
            'ACORN': ['C', 'P', 'F'],
            'MAE': [0.5123, 0.2768, 0.3015],
            'MAPE (%)': [5.00, 4.83, 6.96],
            'MSE': [1.147, 0.399, 0.355],
        })

        st.dataframe(sarimax_metrics_df)

        st.markdown("""
        The SARIMAX model was trained with the following approach:

        1. **Data Preparation**:
          - Endogenous series: daily `Conso_kWh` for each ACORN group.
          - Exogenous regressors: standardized numeric features (weather, calendar flags, etc.) and one-hot encoded categoricals via a fitted ColumnTransformer.
          - Missing days were filled by forward/backward methods to maintain continuity.

        2. **Model Configuration**:
          - We performed a grid search over (p, d, q) × (P, D, Q, s) combinations, selecting the best by lowest AIC.
          - Final orders used:
            - ACORN-C: (2,0,0) × (0,1,1,14)
            - ACORN-P: (1,0,0) × (0,1,1,14)
            - ACORN-F: (2,0,0) × (1,1,0,7)
          - Stationarity and invertibility checks were relaxed (`enforce_stationarity=False`, `enforce_invertibility=False`).

        3. **Training Process**:
          - Fitted once on the full training period for each ACORN group with its selected hyperparameters.
          - Exogenous features were provided at each fitting step to capture weather and calendar effects.

        4. **Forecast Method**:
          - Generated a 30-day forecast in one shot using `get_forecast(steps=30, exog=exog_test_matrix)`.
          - The exogenous matrix for the test horizon was preprocessed with the same ColumnTransformer used during training.

        5. **Results**:
          - R² scores above 0.94 for all groups, indicating strong predictive power.
          - MAPE values under 4%, showing consistently accurate forecasts.
          - ACORN F achieved the lowest MAPE at 3.22%.
        """)

        st.image("src/img/plot1sarimax.png")
        st.image("src/img/plot2sarimax.png")
        st.image("src/img/plot3sarimax.png")

        st.header("Static Forecast vs. Rolling Forecast")

        st.markdown("""
    ## Static SARIMAX Forecast

    ### Building the Exogenous Matrix
    - Fit a `ColumnTransformer` on training exogenous columns (standardize numeric, one-hot category).
    - Transform the 30-day test exogenous features at once to create a (30 × K) array.
    - Call `model.get_forecast(steps=30, exog=exog_test_array)` to obtain all 30 predictions in one batch.
    - Does **not** update exogenous inputs mid-horizon; uses the same precomputed values each day.

    ## Rolling SARIMAX Forecast

    ### Updating Exogenous Inputs Iteratively
    - For each day i in the 30-day forecast:
      1. Take the fitted SARIMAX model (trained on all history up to day i–1).
      2. Provide the single-day exogenous vector for day i (transformed via ColumnTransformer).
      3. Use `model.predict(start=last_train_index + i, end=last_train_index + i, exog=[exog_i])` to forecast one step ahead.
      4. Append the forecasted value to the endogenous series, then refit or update the state if needed.
    - This feedback loop ensures each day's forecast uses the very latest actuals and dynamic exogenous inputs.

    ### Comparison

    - **Static SARIMAX**  
      - Computes all 30 forecasts in one call, using a fixed exogenous matrix for the entire horizon.  
      - Simpler and faster, but cannot adapt to deviations in exogenous patterns that only reveal themselves mid-horizon.

    - **Rolling SARIMAX**  
      - Forecasts one day at a time, feeding each forecast back into future predictions and updating exogenous values each step.  
      - More computationally expensive but tracks evolving weather or calendar flags that may change during the month.

    Because rolling SARIMAX incorporates the latest exogenous conditions and any new observations, it can correct for shifts in weather or demand patterns, typically outperforming the static approach when underlying drivers evolve suddenly.
        """)
        st.image("src/img/plot1sarimax.png")
        st.image("src/img/plot2sarimax.png")



    with model_tab4:
        st.subheader("MLP")
        st.markdown("30-day forecast using MLP model")
        st.markdown(
            """
### 1. Data and Feature Engineering
- **Data Sources**  
  - Daily consumption (`Conso_kWh`) for days [0, t].  
  - Daily weather features (temperature, humidity, wind speed, precipitation, etc.) for days [0, t+30].  
  - ACORN group label (categorical) for each daily record.  
  - Calendar features: day of week, day of year, month, weekend/weekday flag, holiday flag.

- **Combine into One DataFrame**  
  - Merge consumption and historical weather for days ≤ t, partitioned by ACORN.  
  - Append “future” rows for days t+1 through t+30 with weather forecasts; leave `Conso_kWh` blank for those dates.

- **Feature List**  
  - **Numeric**:  
    ```
    temperatureMax, dewPoint, cloudCover, windSpeed, pressure,
    visibility, humidity, uvIndex, temperatureMin, moonPhase,
    dayofweek, month, dayofyear, dayofweek_sin, dayofweek_cos,
    month_sin, month_cos, dayofyear_sin, dayofyear_cos
    ```  
  - **Categorical**:  
    ```
    windBearing, icon, precipType, is_holiday, is_weekend
    ```  
  - **Group Code**:  
    ```
    Acorn
    ```

### 2. Train/Test Split (Per ACORN)
For each ACORN group separately:  
1. **Identify End of Historical Window**  
   - Let t = last date with known `Conso_kWh` for that ACORN.  
2. **Training Set**  
   - All rows with `Date ≤ t`.  
3. **Forecast Set**  
   - Rows with t < Date ≤ t+30 (weather known, consumption missing).

### 3. Preprocessing and Pipeline Construction (Per ACORN)
- **Preprocessing Steps**  
  1. Standardize all numeric features (zero mean, unit variance).  
  2. One-hot encode each categorical feature (`handle_unknown="ignore"`).

- **MLPRegressor Configuration**  
  - Initial architecture: two hidden layers (e.g. 100 → 50 neurons), ReLU activation, Adam solver, small L2 penalty.  
  - Later tuned via Optuna.

- **Pipeline Sequence**  
  1. **StandardScaler** on numeric features.  
  2. **OneHotEncoder** on categorical features.  
  3. **MLPRegressor** with tuned hyperparameters.

  
### 4. Hyperparameter Tuning with Optuna (Per ACORN)
- **Search Space**  
- Hidden layers: 1–3 layers, each with 32–256 neurons.  
- L2 regularization (`alpha`): 1e-6 – 1e-2 (log scale).  
- Learning rate (`learning_rate_init`): 1e-5 – 1e-2 (log scale).  
- Learning rate schedule: {constant, adaptive}.  
- Activation: {relu, tanh}.  
- Solver: {adam, lbfgs}.  
- Batch size: {32, 64, 128}.  
- Tolerance (`tol`): 1e-5 – 1e-3 (log scale).

- **Tuning Steps**  
1. Instantiate `MLPRegressor` with trial’s hyperparameters.  
2. Perform 3-fold time-series cross-validation on the ACORN’s training set, optimizing RMSE.  
3. Record best hyperparameters and cross-validated RMSE.

### 5. Final Training and Evaluation (Per ACORN)
1. **Retrain** on all days ≤ t using best hyperparameters.  
2. **Predict** consumption for days t+1 – t+30.  
3. **Evaluate on Last 30 Historical Days**  
 - RMSE, MAE, MAPE on hold-out block just before t.  
4. **Forecast Submission**  
 - For days t+1 – t+30, report predicted `Conso_kWh`.

### Results on test set
 """
        )
        metrics_mlp_df = pd.DataFrame({
                'ACORN': ['C', 'P', 'F'],
                'MAE': [0.8805, 0.5238, 0.3556],
                'MAPE (%)': [6.0004, 7.3021, 3.3227], 
                'RMSE': [1.1832, 0.6807, 0.4424]
            })
        st.dataframe(metrics_mlp_df)
        st.image("src/img/plot5mlp.png")
        st.image("src/img/plot6mlp.png")
        st.image("src/img/plot7mlp.png")

        st.markdown(
            """
### Ahead predictions for 30 days
"""
        )
        st.image("src/img/plot1mlp.png")
        st.image("src/img/plot2mlp.png")
        st.image("src/img/plot3mlp.png")
        st.image("src/img/plot4mlp.png")

    with model_tab5:
        st.subheader("SVM")
        st.markdown("30-day forecast using SVM model")
        st.markdown(
            """
### 1. Data and Feature Engineering
- **Data Sources**  
  - Daily consumption (`Conso_kWh`) for days [0, t] per ACORN.  
  - Daily weather features (temperature, humidity, wind speed, precipitation, etc.) for days [0, t+30].  
  - ACORN group label (categorical) for each daily record.  
  - Calendar features: day of week, day of year, month, weekend/weekday flag, holiday flag.

- **Combine into One DataFrame**  
  - Merge consumption and historical weather for days ≤ t.  
  - Append “future” rows for days t+1 through t+30 with weather forecasts; leave `Conso_kWh` empty for those dates.

- **Feature List**  
  - **Numeric**:  
    ```
    temperatureMax, dewPoint, cloudCover, windSpeed, pressure,
    visibility, humidity, uvIndex, temperatureMin, moonPhase,
    dayofweek, month, dayofyear, dayofweek_sin, dayofweek_cos,
    month_sin, month_cos, dayofyear_sin, dayofyear_cos
    ```  
  - **Categorical**:  
    ```
    windBearing, icon, precipType, is_holiday, is_weekend
    ```  
  - **Group Code**:  
    ```
    Acorn
    ```

### 2. Train/Test Split (Per ACORN)
For each ACORN group separately:  
1. **Identify End of Historical Window**  
   - Let t = last date with known `Conso_kWh` for that ACORN.  
2. **Training Set**  
   - All rows with `Date ≤ t`.  
3. **Forecast Set**  
   - Rows with t < Date ≤ t+30 (weather known, consumption missing).

### 3. Preprocessing and Pipeline (Per ACORN)
- **Preprocessing Steps**  
  1. Standardize all numeric features (zero mean, unit variance).  
  2. One-hot encode each categorical feature (`handle_unknown="ignore"`).

- **SVR Configuration**  
  - Kernel: RBF (or linear/poly as tuned).  
  - Hyperparameters to tune: C (penalty), ε (epsilon), γ (for RBF), degree & coef0 (for poly).

- **Pipeline Sequence**
  1. **StandardScaler** on numeric features.  
  2. **OneHotEncoder** on categorical features.  
  3. **SVR** with tuned hyperparameters.


### 4. Hyperparameter Tuning with Optuna (Per ACORN)
- **Search Space**  
- `kernel`: {rbf, linear, poly}  
- `C`: 1e-2 – 1e1 (log scale)  
- `epsilon`: 1e-2 – 1.0 (log scale)  
- If kernel ∈ {rbf, poly}: `gamma`: 1e-4 – 1.0 (log scale)  
- If kernel = poly: `degree`: {2, 3, 4}, `coef0`: {0.0, 0.1, 1.0}

- **Tuning Steps**  
1. Instantiate SVR with trial’s hyperparameters.  
2. Perform 3-fold time-series cross-validation on the ACORN’s training set, optimizing RMSE.  
3. Record best hyperparameters and CV RMSE.

### 5. Final Training and Evaluation (Per ACORN)
1. **Retrain** on all days ≤ t using best hyperparameters.  
2. **Predict** consumption for days t+1 – t+30.  
3. **Evaluate on Last 30 Historical Days**  
 - RMSE, MAE, MAPE on hold-out block just before t.  
4. **Forecast Submission**  
 - For days t+1 – t+30, report predicted `Conso_kWh`.
 
### Results on test set
"""
        )
        metrics_svm_df = pd.DataFrame({
            'ACORN': ['C', 'P', 'F'],
                'MAE': [0.8938, 0.4108, 0.3810],
                'MAPE (%)': [5.9040, 5.7737, 3.5947], 
                'RMSE': [1.3435, 0.4723, 0.4846]
        })
        st.dataframe(metrics_svm_df)
        st.image("src/img/plotsvm1.png")
        st.image("src/img/plotsvm2.png")
        st.image("src/img/plotsvm3.png")

        st.markdown(
            """
### Ahead predictions for 30 days
"""
        )
        st.image("src/img/plotsvm4.png")
        st.image("src/img/plotsvm5.png")
        st.image("src/img/plotsvm6.png")
        st.image("src/img/plotsvm7.png")