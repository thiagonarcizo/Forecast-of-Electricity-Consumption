import pandas as pd
# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# PARTIE THIAGO

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Data loading and preprocessing functions
def load_all_data():
    """Load all raw and processed data files"""
    data = {}
    
    # Load raw data
    data['acorn_details'] = pd.read_csv('data/00_raw/acorn_details.csv', encoding='ISO-8859-1')
    data['temperatures'] = pd.read_csv('data/00_raw/temperatures.csv', sep=';', decimal=',', encoding='utf-8')
    data['uk_bank_holidays'] = pd.read_csv('data/00_raw/uk_bank_holidays.csv')
    data['weather_daily'] = pd.read_csv('data/00_raw/weather_daily_darksky.csv')
    data['weather_hourly'] = pd.read_csv('data/00_raw/weather_hourly_darksky.csv')
    
    # Load processed data from parquet
    data['group_4_daily_predict'] = pd.read_parquet('data/02_processed/parquet/group_4_daily_predict.parquet')
    data['group_4_half_hourly_predict'] = pd.read_parquet('data/02_processed/parquet/group_4_half_hourly_predict.parquet')
    data['group_4_daily'] = pd.read_parquet('data/02_processed/parquet/group_4_daily.parquet')
    data['group_4_half_hourly'] = pd.read_parquet('data/02_processed/parquet/group_4_half_hourly.parquet')
    
    return data

def fix_datetime_formats(data):
    """Fix datetime formats for all datasets"""
    # Fix temperatures datetime
    data['temperatures']['DateTime'] = pd.to_datetime(data['temperatures']['DateTime'], format='mixed')
    
    # Fix bank holidays datetime
    data['uk_bank_holidays']['Bank holidays'] = pd.to_datetime(data['uk_bank_holidays']['Bank holidays'], format='mixed')
    
    # Fix weather daily datetime columns
    datetime_columns = ['temperatureMaxTime', 'temperatureMinTime', 'apparentTemperatureMinTime', 
                       'apparentTemperatureHighTime', 'time', 'sunsetTime', 'sunriseTime', 
                       'temperatureHighTime', 'uvIndexTime', 'temperatureLowTime', 
                       'apparentTemperatureMaxTime', 'apparentTemperatureLowTime']
    
    for col in datetime_columns:
        data['weather_daily'][col] = pd.to_datetime(data['weather_daily'][col])
    
    return data

# Seasonal analysis functions
def get_season(month):
    """Convert month number to season name"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # months 9, 10, 11
        return 'Fall'

def add_temporal_features(df, datetime_col='DateTime'):
    """Add temporal features to dataframe"""
    df = df.copy()
    df['Hour'] = df[datetime_col].dt.hour
    df['Day'] = df[datetime_col].dt.day_name()
    df['Date'] = df[datetime_col].dt.date
    df['Season'] = df[datetime_col].dt.month.apply(get_season)
    return df

# Plotting utility functions
def create_boxplot_by_acorn(data, y_col='Conso_moy', title_prefix='', figsize=(12, 6)):
    """Create boxplot comparing Acorn groups"""
    plt.figure(figsize=figsize)
    sns.boxplot(x='Acorn', y=y_col, data=data, palette='Set2')
    plt.title(f'{title_prefix} Consumption by Acorn Group')
    plt.xlabel('Acorn Group')
    plt.ylabel('Consumption (kWh)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def create_heatmap_by_acorn(data, acorn_groups, day_order, figsize=(20, 6)):
    """Create heatmaps for each Acorn group showing hour vs day patterns"""
    fig, axes = plt.subplots(1, len(acorn_groups), figsize=figsize)
    
    # Calculate global min and max for consistent color scale
    all_means = []
    for acorn in acorn_groups:
        acorn_data = data[data['Acorn'] == acorn]
        heatmap_data = acorn_data.pivot_table(index='Hour', columns='Day', values='Conso_moy', aggfunc='mean')
        all_means.extend(heatmap_data.values.flatten())
    
    # Remove NaN values and calculate global range
    all_means = [x for x in all_means if not pd.isna(x)]
    vmin, vmax = min(all_means), max(all_means)
    
    for idx, acorn in enumerate(acorn_groups):
        # Filter data for current Acorn group
        acorn_data = data[data['Acorn'] == acorn]
        
        # Create pivot table for heatmap
        heatmap_data = acorn_data.pivot_table(index='Hour', columns='Day', values='Conso_moy', aggfunc='mean')
        heatmap_data = heatmap_data.reindex(columns=day_order)
        
        # Create heatmap with consistent color scale
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', 
                    cbar_kws={'label': 'Average Consumption (kWh)'}, ax=axes[idx],
                    vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'Consumption Heatmap - {acorn}')
        axes[idx].set_xlabel('Day of the Week')
        axes[idx].set_ylabel('Hour of the Day')
    
    plt.tight_layout()
    plt.show()

def create_load_duration_curves(data, acorn_groups, figsize=(18, 6)):
    """Create load duration curves for each Acorn group"""
    fig, axes = plt.subplots(1, len(acorn_groups), figsize=figsize)
    
    for idx, acorn_group in enumerate(acorn_groups):
        # Filter data for current Acorn group
        acorn_data = data[data['Acorn'] == acorn_group]
        
        # Sort consumption values in descending order
        sorted_consumption = acorn_data['Conso_moy'].sort_values(ascending=False).reset_index(drop=True)
        
        # Create time duration as percentage (0 to 100%)
        duration_percent = (sorted_consumption.index / len(sorted_consumption)) * 100
        
        # Plot load duration curve
        axes[idx].plot(duration_percent, sorted_consumption, linewidth=2, color='blue')
        axes[idx].set_title(f'Load Duration Curve - {acorn_group}')
        axes[idx].set_xlabel('Duration (%)')
        axes[idx].set_ylabel('Consumption (kWh)')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()

def create_temporal_boxplots(data, acorn_groups, day_order, figsize=(20, 12)):
    """Create boxplots showing temporal patterns for each Acorn group"""
    fig, axes = plt.subplots(2, len(acorn_groups), figsize=figsize)
    
    for idx, acorn_group in enumerate(acorn_groups):
        acorn_data = data[data['Acorn'] == acorn_group]
        
        # Hour boxplot
        sns.boxplot(x='Hour', y='Conso_moy', data=acorn_data, ax=axes[0, idx], 
                   hue='Hour', palette='viridis', legend=False)
        axes[0, idx].set_title(f'Consumption by Hour - {acorn_group}')
        axes[0, idx].set_xlabel('Hour of Day')
        axes[0, idx].set_ylabel('Consumption (kWh)')
        axes[0, idx].tick_params(axis='x', rotation=45)
        
        # Day of week boxplot
        sns.boxplot(x='Day', y='Conso_moy', data=acorn_data, ax=axes[1, idx], 
                   order=day_order, hue='Day', palette='Set2', legend=False)
        axes[1, idx].set_title(f'Consumption by Day - {acorn_group}')
        axes[1, idx].set_xlabel('Day of Week')
        axes[1, idx].set_ylabel('Consumption (kWh)')
        axes[1, idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_seasonal_analysis(data, acorn_groups, season_order, figsize=(16, 20)):
    """Create seasonal subseries analysis plots"""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    for idx, season in enumerate(season_order):
        season_data = data[data['Season'] == season]
        
        for acorn_group in acorn_groups:
            acorn_season_data = season_data[season_data['Acorn'] == acorn_group]
            
            if len(acorn_season_data) > 0:
                # Calculate hourly means for this season and Acorn group
                hourly_means = acorn_season_data.groupby('Hour')['Conso_moy'].mean()
                hourly_std = acorn_season_data.groupby('Hour')['Conso_moy'].std()
                
                # Plot the hourly pattern
                axes[idx].plot(hourly_means.index, hourly_means.values, 
                              marker='o', linewidth=2, markersize=4, 
                              label=acorn_group, alpha=0.8)
                
                # Add confidence intervals
                axes[idx].fill_between(hourly_means.index,
                                      hourly_means.values - hourly_std.values,
                                      hourly_means.values + hourly_std.values,
                                      alpha=0.2)
        
        # Customize each subplot
        axes[idx].set_title(f'{season} - Hourly Consumption Patterns', 
                           fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Average Consumption (kWh)', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(title='Acorn Group', loc='upper right')
        axes[idx].set_xticks(range(0, 24, 2))
        axes[idx].set_xlim(0, 23)
    
    axes[3].set_xlabel('Hour of Day', fontsize=12)
    plt.suptitle('Seasonal Subseries: Hourly Consumption Patterns by Season', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

def print_seasonal_summary(data, acorn_groups, season_order):
    """Print seasonal summary statistics"""
    print("\n=== SEASONAL SUBSERIES ANALYSIS ===")
    print("Peak hours and consumption levels by season and Acorn group:")
    
    for season in season_order:
        print(f"\n--- {season.upper()} ---")
        season_data = data[data['Season'] == season]
        
        for acorn_group in acorn_groups:
            acorn_season_data = season_data[season_data['Acorn'] == acorn_group]
            
            if len(acorn_season_data) > 0:
                hourly_means = acorn_season_data.groupby('Hour')['Conso_moy'].mean()
                
                # Find peak and minimum hours
                peak_hour = hourly_means.idxmax()
                min_hour = hourly_means.idxmin()
                peak_consumption = hourly_means.max()
                min_consumption = hourly_means.min()
                
                print(f"{acorn_group}: Peak at {peak_hour:02d}:00 ({peak_consumption:.3f} kWh), "
                      f"Min at {min_hour:02d}:00 ({min_consumption:.3f} kWh)")

# Function to perform statistical tests
def perform_holiday_statistical_analysis(data, acorn_groups, consumption_col='Conso_kWh'):
    """Perform statistical tests comparing holidays vs weekends and holidays vs weekdays"""
    from scipy import stats
    
    results = []
    
    for acorn_group in acorn_groups:
        acorn_data = data[data['Acorn'] == acorn_group]
        
        holiday_consumption = acorn_data[acorn_data['Day_Category'] == 'Holiday'][consumption_col]
        weekend_consumption = acorn_data[acorn_data['Day_Category'] == 'Weekend'][consumption_col]
        weekday_consumption = acorn_data[acorn_data['Day_Category'] == 'Weekday'][consumption_col]
        
        # Holiday vs Weekend
        if len(holiday_consumption) > 0 and len(weekend_consumption) > 0:
            t_stat_hw, p_value_hw = stats.ttest_ind(holiday_consumption, weekend_consumption)
            pooled_std_hw = np.sqrt(((len(holiday_consumption) - 1) * holiday_consumption.var() + 
                                     (len(weekend_consumption) - 1) * weekend_consumption.var()) / 
                                    (len(holiday_consumption) + len(weekend_consumption) - 2))
            cohens_d_hw = (holiday_consumption.mean() - weekend_consumption.mean()) / pooled_std_hw if pooled_std_hw else 0
            
            results.append({
                'Comparison': 'Holiday vs Weekend',
                'Acorn_Group': acorn_group,
                'Group1_Mean': holiday_consumption.mean(),
                'Group1_Std': holiday_consumption.std(),
                'Group1_Count': len(holiday_consumption),
                'Group2_Mean': weekend_consumption.mean(),
                'Group2_Std': weekend_consumption.std(),
                'Group2_Count': len(weekend_consumption),
                'Mean_Difference': holiday_consumption.mean() - weekend_consumption.mean(),
                'T_Statistic': t_stat_hw,
                'P_Value': p_value_hw,
                'Cohens_D': cohens_d_hw,
                'Significant': p_value_hw < 0.05
            })

        # Holiday vs Weekday
        if len(holiday_consumption) > 0 and len(weekday_consumption) > 0:
            t_stat_hd, p_value_hd = stats.ttest_ind(holiday_consumption, weekday_consumption)
            pooled_std_hd = np.sqrt(((len(holiday_consumption) - 1) * holiday_consumption.var() + 
                                     (len(weekday_consumption) - 1) * weekday_consumption.var()) / 
                                    (len(holiday_consumption) + len(weekday_consumption) - 2))
            cohens_d_hd = (holiday_consumption.mean() - weekday_consumption.mean()) / pooled_std_hd if pooled_std_hd else 0
            
            results.append({
                'Comparison': 'Holiday vs Weekday',
                'Acorn_Group': acorn_group,
                'Group1_Mean': holiday_consumption.mean(),
                'Group1_Std': holiday_consumption.std(),
                'Group1_Count': len(holiday_consumption),
                'Group2_Mean': weekday_consumption.mean(),
                'Group2_Std': weekday_consumption.std(),
                'Group2_Count': len(weekday_consumption),
                'Mean_Difference': holiday_consumption.mean() - weekday_consumption.mean(),
                'T_Statistic': t_stat_hd,
                'P_Value': p_value_hd,
                'Cohens_D': cohens_d_hd,
                'Significant': p_value_hd < 0.05
            })
            
    return pd.DataFrame(results)

############################################################################################################

# Global constants
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']

#######################################

def load_weather_data() -> pd.DataFrame:

    """
    Load and preprocess the weather data from a CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with 'date' as the index.
    """
    df = pd.read_csv(r'data\00_raw\weather_daily_darksky.csv')
    df = preprocess_weather_daily(df)

    return df
