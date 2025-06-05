import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.streamlit_helpers import *

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
)

st.title("📊 Exploratory Data Analysis")

st.markdown("This page contains the Exploratory Data Analysis of the project.")

# Load data
data = load_all_data()
daily_data = data['group_4_daily']
half_hourly_data = data['group_4_half_hourly']
uk_bank_holidays = data['uk_bank_holidays']

# Add temporal features
half_hourly_data['DateTime'] = pd.to_datetime(half_hourly_data['DateTime'])
half_hourly_data = add_temporal_features(half_hourly_data, 'DateTime')

# Acorn groups
acorn_groups = half_hourly_data['Acorn'].unique()
acorn_selection = st.multiselect("Select Acorn Groups to Compare (up to 3)", acorn_groups, default=[acorn_groups[0]] if acorn_groups.size > 0 else [])

if not acorn_selection:
    st.warning("Please select at least one Acorn group.")
elif len(acorn_selection) > 3:
    st.warning("Please select no more than three Acorn groups to compare.")
else:
    # Filter data for selected Acorn group
    daily_data_acorn = daily_data[daily_data['Acorn'].isin(acorn_selection)]
    half_hourly_data_acorn = half_hourly_data[half_hourly_data['Acorn'].isin(acorn_selection)]

    # Display plots
    st.header(f"Analysis for Acorn Group(s): {', '.join(acorn_selection)}")

    st.subheader("Consumption Heatmap")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig = create_heatmap_by_acorn(half_hourly_data_acorn, acorn_selection, day_order)
    st.pyplot(fig)

    st.subheader("Load Duration Curve")
    fig = create_load_duration_curves(half_hourly_data_acorn, acorn_selection)
    st.pyplot(fig)

    st.subheader("Seasonal Analysis")
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    fig = create_seasonal_analysis(half_hourly_data_acorn, acorn_selection, season_order)
    st.pyplot(fig)

    st.subheader("Seasonal Summary")
    summary = print_seasonal_summary(half_hourly_data_acorn, acorn_selection, season_order)
    st.text(summary)

    st.subheader("Daily, Weekly, Monthly, and Seasonal Consumption")
    fig1, fig2, fig3, fig4, fig5, fig6 = plot_daily_acorn_consumption(daily_data_acorn, uk_bank_holidays)
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)
    st.pyplot(fig5)
    st.pyplot(fig6)

    st.subheader("Outlier Boxplots")
    fig1, fig2, fig3, fig4 = plot_daily_acorn_outlier_boxplots(daily_data_acorn, uk_bank_holidays=uk_bank_holidays)
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4) 

    st.header("Holiday Analysis")
    st.markdown('''
                ### Holiday Electricity Consumption Analysis  
### 1 · Group-by-Group Findings

| Comparison | ACORN-C | ACORN-F | ACORN-P |
|------------|---------|---------|---------|
| **Holiday vs Weekend** | **+0.89 kWh** (small effect: d = 0.44); *not sig.* (p = 0.11) | –0.09 kWh (negligible: d = –0.05); *not sig.* | –0.01 kWh (negligible: d = –0.01); *not sig.* |
| **Holiday vs Weekday** | **+1.42 kWh** (medium effect: d = 0.79); **significant** (p = 0.004) | +0.45 kWh (small effect: d = 0.32); *not sig.* | +0.15 kWh (negligible: d = 0.13); *not sig.* |

**All significance tests are two-tailed; d = Cohen’s d.**

---

### 2 · Interpretation

1. **ACORN-C (higher-consumption households)**  
   * Holidays show **meaningfully higher** usage than both weekends (+7 %) and weekdays (+12 %).  
   * Only the holiday-vs-weekday jump is statistically reliable (medium-sized, p < 0.01).  
   * Implication: special-occasion behaviour in this segment translates into real extra load.

2. **ACORN-F (mid-consumption households)**  
   * Holidays look virtually identical to weekends (–0.9 %).  
   * They run **about 5 % above weekdays**, but the effect is small and not significant.  
   * Behaviour suggests a mild “leisure-day” boost that is lost in statistical noise.

3. **ACORN-P (lower-consumption households)**  
   * Differences are practically zero versus both weekends and weekdays.  
   * Consumption patterns appear stable regardless of day type.

---

### 3 · Cross-Group Patterns

| Key Question | Short Answer |
|--------------|--------------|
| **Do holidays always beat weekends?** | Only for ACORN-C. Groups F and P actually dip (slightly) on holidays. |
| **Do holidays beat weekdays?** | Yes in every group, but the jump is **material and statistically solid only for ACORN-C**. |
| **Where is the practical impact?** | Aggregate load forecasting should flag holiday demand spikes mainly in areas with many ACORN-C customers. |

---

### 4 · Practical Takeaways

* **Targeted interventions** (e.g., holiday energy-saving campaigns) should focus on ACORN-C households; the other groups won’t yield big returns.  
* **System-level forecasting** should add a holiday premium chiefly when the network has a high share of ACORN-C customers.  
* **Statistical nuance:** absence of significance in F and P means observed differences could be pure sampling noise—avoid over-interpreting tiny kWh shifts.  
                ''')
    
    st.header("External Influence (Forecast)")
    # Load necessary data
    data = load_all_data()
    weather_hourly = data['weather_hourly']
    group_4_half_hourly = data['group_4_half_hourly']

    # Reset index to make time a column if it's currently the index
    if weather_hourly.index.name == 'time' or 'time' in str(weather_hourly.index.name):
        weather_hourly = weather_hourly.reset_index()

    # Ensure datetime columns are properly formatted
    if 'time' in weather_hourly.columns and not pd.api.types.is_datetime64_any_dtype(weather_hourly['time']):
        weather_hourly['time'] = pd.to_datetime(weather_hourly['time'])
    elif 'DateTime' in weather_hourly.columns and not pd.api.types.is_datetime64_any_dtype(weather_hourly['DateTime']):
        weather_hourly['DateTime'] = pd.to_datetime(weather_hourly['DateTime'])

    # Resample group_4_half_hourly to hourly frequency
    group_4_hourly = group_4_half_hourly.set_index('DateTime').resample('h')['Conso_moy'].mean().reset_index()

    # Determine the correct timestamp column in weather_hourly for merging
    weather_time_col = 'time' if 'time' in weather_hourly.columns else 'DateTime'

    # Merge with weather_hourly data
    merged_data = pd.merge(group_4_hourly, weather_hourly, left_on='DateTime', right_on=weather_time_col, how='inner')

    # Select numerical columns for correlation
    columns_to_exclude = ['block_id', 'LCLid']
    relevant_numeric_cols = [col for col in merged_data.select_dtypes(include=np.number).columns if col not in columns_to_exclude]
    correlation_matrix = merged_data[relevant_numeric_cols].corr()

    # Display the correlation matrix using a heatmap
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix: Hourly Consumption vs Weather Data', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('''
## 1 · Instantaneous Relationships (no lag)

| Weather driver | Corr. with consumption | Reading |
|----------------|-----------------------|---------|
| Temperature / Apparent T° | **~ –0.31 / –0.33** | Moderate, negative |
| Dew-point      | –0.36 | Moderate, negative |
| Wind-speed     | +0.16 | Weak, positive |
| Visibility, Wind-bearing, Pressure, Humidity | ~0 | Essentially no linear link |

**Take-away:**  
Higher outdoor temperatures (and thus higher dew-point / apparent-T°) tend to **reduce** demand; the effect is sizeable but not overwhelming.
Wind has a small opposite effect – when it is windier, people use slightly more energy (likely space-heating losses). Most other weather variables show negligible instantaneous influence.
                ''')
    
    st.header("2 · Short-Lag Effects (1-6 hours)")

    # Create merged_data_lag for lag analysis
    merged_data_lag = merged_data.set_index('DateTime')

    # Weather columns to lag
    weather_cols_for_lag = ['temperature', 'humidity', 'windSpeed', 'apparentTemperature', 'dewPoint', 'pressure']

    lag_periods = range(1, 7)  # Lags from 1 to 6 hours

    lag_correlations = {}

    for weather_col in weather_cols_for_lag:
        corrs = []
        for lag in lag_periods:
            lagged_col_name = f'{weather_col}_lag{lag}h'
            merged_data_lag[lagged_col_name] = merged_data_lag[weather_col].shift(lag)
            correlation = merged_data_lag['Conso_moy'].corr(merged_data_lag[lagged_col_name])
            corrs.append(correlation)
        lag_correlations[weather_col] = corrs

    # Create a DataFrame for easier plotting
    lag_corr_df = pd.DataFrame(lag_correlations, index=[f'{lag}h lag' for lag in lag_periods])

    # Plotting the lag correlations
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(lag_corr_df.T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Lag Correlation: Consumption vs. Weather Variables (1-6 hour lags)', fontsize=16)
    plt.ylabel('Weather Variable')
    plt.xlabel('Time Lag')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    # Plot lag correlation for temperature
    fig = plt.figure(figsize=(9, 5))
    lag_corr_df['temperature'].plot(kind='bar', color='skyblue')
    plt.title('Lag Correlation: Consumption vs. Temperature', fontsize=14)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Time Lag')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('''
| Variable | 1 h-lag | 3 h-lag | 6 h-lag | Pattern |
|----------|---------|---------|---------|---------|
| Temperature | –0.29 | –0.27 | **–0.31** | Stable, slightly stronger again after 6 h |
| Apparent T° | –0.32 | –0.30 | –0.33 | Mirrors temperature |
| Dew-point   | –0.36 | –0.37 | –0.37 | Consistently strongest negative driver |
| Wind-speed  | +0.18 | +0.23 | **+0.23** | Small but growing positive effect, peaks ≈ 5 h |
| Humidity    | –0.06 | –0.11 | –0.05 | Very weak, noisy |
| Pressure    | –0.11 | –0.11 | –0.11 | Flat, weak negative |

**Take-away:** 

* The demand reaction to **temperature-related variables is immediate and lasts for several hours**; there is no evidence that the impact fades quickly.  
* **Wind-speed has its maximal influence about 4-5 h later**, hinting that wind-driven heat loss (or perceived cold) takes a few hours to translate into higher electricity use (thermostat response, occupant behaviour).  
* Other signals remain close to the noise floor.
                ''')
    
    st.header("3 · Long-Lag Effects (1-7 days)")
    # Merge daily consumption and weather data
    weather_daily_with_date = data['weather_daily'].reset_index()
    weather_daily_with_date['Date'] = pd.to_datetime(weather_daily_with_date['time']).dt.date
    merged_daily = merge_daily_consumption_weather(daily_data, weather_daily_with_date)

    # Weather columns for long lag analysis
    weather_cols_daily = ['temperatureMax', 'temperatureMin', 'humidity', 'windSpeed', 
                         'pressure', 'cloudCover', 'apparentTemperatureMax', 'apparentTemperatureMin']
    weather_cols_daily = [col for col in weather_cols_daily if col in merged_daily.columns]

    # Set Date as index for lag analysis
    merged_daily_lag = merged_daily.set_index('Date').sort_index()

    # Calculate correlations for 1-7 day lags
    long_lag_periods = range(1, 8)
    long_lag_correlations = {}

    for weather_col in weather_cols_daily:
        corrs = []
        for lag in long_lag_periods:
            lagged_col_name = f'{weather_col}_lag{lag}d'
            merged_daily_lag[lagged_col_name] = merged_daily_lag[weather_col].shift(lag)
            correlation = merged_daily_lag['Conso_kWh'].corr(merged_daily_lag[lagged_col_name])
            if pd.isna(correlation):
                correlation = 0.0
            corrs.append(correlation)
        long_lag_correlations[weather_col] = corrs

    long_lag_corr_df = pd.DataFrame(long_lag_correlations, 
                                   index=[f'{lag}d lag' for lag in long_lag_periods])

    # Plot heatmap of long-lag correlations
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(long_lag_corr_df.T, annot=True, cmap='coolwarm', fmt=".3f", 
                linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Long Lag Cross-Correlation: Daily Consumption vs. Weather Variables (1-7 day lags)', 
             fontsize=16, fontweight='bold')
    plt.ylabel('Weather Variable', fontsize=12)
    plt.xlabel('Time Lag (Days)', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    # Plot detailed analysis for key variables
    vars_to_plot = weather_cols_daily[:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['skyblue', 'lightgreen', 'salmon', 'plum']

    for i, weather_var in enumerate(vars_to_plot):
        if i < len(axes) and weather_var in long_lag_corr_df.columns:
            ax = axes[i]
            correlations = long_lag_corr_df[weather_var]
            
            bars = ax.bar(range(len(correlations)), correlations.values,
                         color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            max_idx = np.abs(correlations.values).argmax()
            bars[max_idx].set_color('red')
            bars[max_idx].set_alpha(0.9)
            
            ax.set_title(f'Lag Correlation: {weather_var}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Days Lag', fontsize=10)
            ax.set_ylabel('Correlation Coefficient', fontsize=10)
            ax.set_xticks(range(len(correlations)))
            ax.set_xticklabels([f'{lag}d' for lag in long_lag_periods])
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001 * np.sign(height),
                       f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)

    plt.suptitle('Long Lag Cross-Correlation Profiles by Weather Variable',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown('''
| Weather driver | Peak lag & sign | Strength | Comment |
|----------------|-----------------|----------|---------|
| **Max temperature** | **2 days, –0.499** | Strong | Cooler spells lift demand for ~a week. |
| Min temperature | 1 day, –0.463 | Strong | Similar but slightly weaker. |
| Apparent T° (max/min) | 2 days, –0.50 | Strong | Mirrors physical temperature. |
| **Humidity** | 5 days, +0.255 | Moderate | More humid periods coincide with higher consumption a few days later (possible proxy for rainy/cloudy weather --> indoor lighting / heating). |
| Cloud-cover | 1–7 days, +0.16-0.18 | Weak-to-moderate | Same intuition as humidity. |
| Wind-speed | 1 day, +0.111 | Weak | Daily average effect is modest. |
| Pressure | 5-7 days, –0.125 | Weak | Low-pressure systems (bad weather) slightly increase demand. |

**Take-away:**

* **Temperature is the dominant long-range predictor**: a cold snap drives up electricity use for several consecutive days.  
* **Humidity and cloud-cover show the opposite sign** (positive): dull, damp weather tends to keep people indoors with lights/heating on.  
* **Wind and pressure matter, but only marginally** once we aggregate to daily totals.

---

## 4 · Putting It All Together

1. **Heating-dominated profile** – The consistently **negative temperature correlations** (hourly and daily) imply that electricity demand rises as the air gets colder, typical of a heating-centric load (electric radiators, heat pumps or resistive heaters).  
2. **Lag behaviour** – Immediate (0–1 h) and sustained (up to 6 h and multiple days) effects indicate that both quick thermostat responses and prolonged cold spells shape consumption patterns.  
3. **Weather system influence** – Positive links with humidity/cloud-cover and small negatives with pressure show that *broader weather regimes* (e.g., overcast, low-pressure systems) also raise demand, independent of raw temperature.  
4. **Wind sensitivity** – Although weaker than temperature, wind contributes a **small incremental load**; design of energy-efficiency measures should consider infiltration and draught proofing.
5. **Recap of the interpretation of correlation** - Strong: $|r| > 0.7$, Moderate: $0.3 < |r| ≤ 0.7$, Weak: $0.1 < |r| ≤ 0.3$, Negligible: $|r| ≤ 0.1$
                ''')