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
    data['weather_daily'] = pd.read_parquet('data/01_interim/weather_daily_darksky_cleaned.parquet')
    data['weather_hourly'] = pd.read_parquet('data/01_interim/weather_hourly_darksky_cleaned.parquet')
    
    # Load processed data from parquet
    data['group_4_daily_predict'] = pd.read_parquet('data/02_processed/parquet/group_4_daily_predict.parquet')
    data['group_4_half_hourly_predict'] = pd.read_parquet('data/02_processed/parquet/group_4_half_hourly_predict.parquet')
    data['group_4_daily'] = pd.read_parquet('data/02_processed/parquet/group_4_daily.parquet')
    data['group_4_half_hourly'] = pd.read_parquet('data/02_processed/parquet/group_4_half_hourly.parquet')
    
    return data

def fix_weather_daily_date(weather_daily):
    """
    Ensures weather_daily has a 'Date' column of type datetime.date, handling both DatetimeIndex and column cases.
    """
    df = weather_daily.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if 'Date' not in df.columns:
        # Try to find a date-like column
        for col in df.columns:
            if 'date' in col.lower():
                df['Date'] = pd.to_datetime(df[col]).dt.date
                break
        else:
            raise ValueError("No date column found in weather_daily DataFrame.")
    else:
        df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def merge_daily_consumption_weather(consumption_daily, weather_daily, consumption_date_col='Date', weather_date_col='Date'):
    """
    Merges daily consumption and weather DataFrames on the date, using fix_weather_daily_date for robust handling.
    """
    # Prepare consumption data
    df_conso = consumption_daily.copy()
    if consumption_date_col not in df_conso.columns:
        # Try to find a date-like column
        for col in df_conso.columns:
            if 'date' in col.lower():
                df_conso['Date'] = pd.to_datetime(df_conso[col]).dt.date
                break
        else:
            raise ValueError("No date column found in consumption_daily DataFrame.")
    else:
        df_conso['Date'] = pd.to_datetime(df_conso[consumption_date_col]).dt.date
    # Prepare weather data
    df_weather = fix_weather_daily_date(weather_daily)
    # Merge
    merged = pd.merge(df_conso, df_weather, left_on='Date', right_on='Date', how='inner')
    return merged

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
    fig = plt.figure(figsize=figsize)
    sns.boxplot(x='Acorn', y=y_col, data=data, palette='Set2')
    plt.title(f'{title_prefix} Consumption by Acorn Group')
    plt.xlabel('Acorn Group')
    plt.ylabel('Consumption (kWh)')
    plt.xticks(rotation=45)
    return fig

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
    
    return fig

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
    
    return fig

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
    
    return fig

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
    
    return fig

def print_seasonal_summary(data, acorn_groups, season_order):
    """Print seasonal summary statistics"""
    summary = []
    summary.append("\n=== SEASONAL SUBSERIES ANALYSIS ===")
    summary.append("Peak hours and consumption levels by season and Acorn group:")
    
    for season in season_order:
        summary.append(f"\n--- {season.upper()} ---")
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
                
                summary.append(f"{acorn_group}: Peak at {peak_hour:02d}:00 ({peak_consumption:.3f} kWh), "
                      f"Min at {min_hour:02d}:00 ({min_consumption:.3f} kWh)")
    return "\n".join(summary)

def plot_daily_acorn_consumption(daily, uk_bank_holidays, acorn_types=None):
    """Plot daily, weekly, monthly, and seasonal consumption for each Acorn group, with holidays highlighted."""
    import matplotlib.pyplot as plt
    import pandas as pd
    if acorn_types is None:
        acorn_types = daily['Acorn'].unique()
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(daily['Date']):
        daily['Date'] = pd.to_datetime(daily['Date'])
    acorn_data = {acorn: daily[daily['Acorn'] == acorn].reset_index(drop=True) for acorn in acorn_types}
    # Daily average
    daily_avg_consumption = {acorn: df[['Date', 'Conso_kWh']].reset_index(drop=True) for acorn, df in acorn_data.items()}
    bank_holiday_dates = pd.to_datetime(uk_bank_holidays['Bank holidays']).dt.date
    fig = plt.figure(figsize=(12, 8))
    for acorn, daily_avg in daily_avg_consumption.items():
        plt.plot(daily_avg['Date'], daily_avg['Conso_kWh'], label=acorn)
        bh_dates = []
        bh_conso = []
        for date in sorted(bank_holiday_dates):
            if date in daily_avg['Date'].dt.date.values:
                value = daily_avg.loc[daily_avg['Date'].dt.date == date, 'Conso_kWh'].values
                if len(value) > 0:
                    bh_dates.append(date)
                    bh_conso.append(value[0])
        plt.scatter(bh_dates, bh_conso, color='red', s=10, zorder=5)
    plt.title('Daily Average Consumption by Acorn Type\n(Red dots: UK Bank Holidays)')
    plt.xlabel('Date')
    plt.ylabel('Average Consumption (kWh)')
    plt.legend()
    plt.grid()
    
    # By weekday
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg_by_weekday = {}
    for acorn, data in acorn_data.items():
        data = data.copy()
        data['Weekday'] = data['Date'].dt.dayofweek
        avg_by_weekday = data.groupby('Weekday', as_index=False)['Conso_kWh'].mean()
        avg_by_weekday['Day'] = avg_by_weekday['Weekday'].map(dict(enumerate(day_names)))
        daily_avg_by_weekday[acorn] = avg_by_weekday
    fig2 = plt.figure(figsize=(10, 6))
    for acorn, avg_by_weekday in daily_avg_by_weekday.items():
        plt.plot(avg_by_weekday['Day'], avg_by_weekday['Conso_kWh'], label=acorn, marker='o')
    plt.title('Average Consumption by Day of the Week (All Year)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Consumption (kWh)')
    plt.legend()
    plt.grid()
    
    # By weekday and season
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig3, axes = plt.subplots(1, len(acorn_data), figsize=(20, 6), sharey=True)
    for ax, (acorn, data) in zip(axes, acorn_data.items()):
        data = data.copy()
        data['Season'] = pd.cut(data['Date'].dt.month, bins=[0, 3, 6, 9, 12], labels=seasons, right=False)
        data['Weekday'] = data['Date'].dt.dayofweek
        for season in seasons:
            season_data = data[data['Season'] == season]
            avg_by_weekday = season_data.groupby('Weekday', as_index=False)['Conso_kWh'].mean()
            avg_by_weekday['Day'] = avg_by_weekday['Weekday'].map(dict(enumerate(day_names)))
            ax.plot(avg_by_weekday['Day'], avg_by_weekday['Conso_kWh'], label=season, marker='o')
        ax.set_title(f'{acorn}')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Average Consumption (kWh)')
        ax.legend()
        ax.grid()
    
    # Weekly
    weekly_avg_consumption = {}
    for acorn, data in acorn_data.items():
        data = data.copy()
        data['Week'] = data['Date'].dt.isocalendar().week
        weekly_avg = data.groupby('Week', as_index=False)['Conso_kWh'].mean()
        weekly_avg_consumption[acorn] = weekly_avg
    fig4 = plt.figure(figsize=(12, 8))
    for acorn, weekly_avg in weekly_avg_consumption.items():
        plt.plot(weekly_avg['Week'], weekly_avg['Conso_kWh'], label=acorn)
    plt.title('Weekly Average Consumption by Acorn Type')
    plt.xlabel('Week of the Year')
    plt.ylabel('Average Consumption (kWh)')
    plt.legend()
    plt.grid()
    
    # Seasonal
    seasonal_avg_consumption = {}
    for acorn, data in acorn_data.items():
        data = data.copy()
        data['Season'] = pd.cut(data['Date'].dt.month, bins=[0, 3, 6, 9, 12], labels=seasons, right=False)
        seasonal_avg = data.groupby('Season', as_index=False, observed=False)['Conso_kWh'].mean()
        seasonal_avg_consumption[acorn] = seasonal_avg
    fig5 = plt.figure(figsize=(12, 8))
    for acorn, seasonal_avg in seasonal_avg_consumption.items():
        plt.plot(seasonal_avg['Season'], seasonal_avg['Conso_kWh'], label=acorn, marker='o')
    plt.title('Seasonal Average Consumption by Acorn Type')
    plt.xlabel('Season of the Year')
    plt.ylabel('Average Consumption (kWh)')
    plt.legend()
    plt.grid()
    
    # Monthly
    monthly_avg_consumption = {}
    for acorn, data in acorn_data.items():
        data = data.copy()
        data['Month'] = data['Date'].dt.month
        monthly_avg = data.groupby('Month', as_index=False)['Conso_kWh'].mean()
        monthly_avg_consumption[acorn] = monthly_avg
    fig6 = plt.figure(figsize=(12, 8))
    for acorn, monthly_avg in monthly_avg_consumption.items():
        plt.plot(monthly_avg['Month'], monthly_avg['Conso_kWh'], label=acorn, marker='o')
    plt.title('Monthly Average Consumption by Acorn Type')
    plt.xlabel('Month of the Year')
    plt.ylabel('Average Consumption (kWh)')
    plt.legend()
    plt.grid()
    
    return fig, fig2, fig3, fig4, fig5, fig6

def plot_daily_acorn_outlier_boxplots(daily, acorn_data=None, uk_bank_holidays=None):
    """Plot boxplots of daily consumption by Acorn group, day of week, season, and month, overlaying outliers and highlighting bank holidays.
    
    Parameters:
    -----------
    daily : DataFrame
        Daily consumption data with at least Date, Acorn, and Conso_kWh columns
    acorn_data : dict or DataFrame, optional
        If dict: mapping of Acorn groups to DataFrames with consumption data
        If DataFrame: data with Acorn column and consumption data
        If None: will use daily parameter for all calculations
    uk_bank_holidays : DataFrame, optional
        DataFrame with 'Bank holidays' column of holiday dates
        
    Notes:
    ------
    This function handles various input formats and performs datetime validation
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Helper function to get outlier indices
    def get_outlier_indices(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return series[(series < lower) | (series > upper)].index
    
    # Helper function to safely convert Series to DataFrame
    def series_to_df(series, name=None):
        if name is None and hasattr(series, 'name'):
            name = series.name
        if name is None:
            name = 'Conso_kWh'
        return pd.DataFrame({name: series})
    
    # Helper function for checking holiday dates - handles various formats
    def is_holiday(date_series, holiday_dates_set, holiday_dates_list):
        """Check if dates in a series are holidays, handling different date formats"""
        # If we don't have any holiday dates, return False for everything
        if not holiday_dates_set and not holiday_dates_list:
            return pd.Series([False] * len(date_series), index=date_series.index)
        
        results = pd.Series([False] * len(date_series), index=date_series.index)
        
        try:
            # Convert dates to strings for more reliable matching (YYYY-MM-DD format)
            holiday_dates_str = set()
            
            # Convert date objects to strings
            if holiday_dates_set:
                for date_obj in holiday_dates_set:
                    if hasattr(date_obj, 'strftime'):  # It's a date object
                        holiday_dates_str.add(date_obj.strftime('%Y-%m-%d'))
                    elif isinstance(date_obj, str):     # It's already a string
                        holiday_dates_str.add(date_obj)
                        
            # Convert datetime objects to strings
            if holiday_dates_list:
                for dt_obj in holiday_dates_list:
                    if hasattr(dt_obj, 'strftime'):  # It's a datetime object
                        holiday_dates_str.add(dt_obj.strftime('%Y-%m-%d'))
                    elif isinstance(dt_obj, str):     # It's already a string
                        holiday_dates_str.add(dt_obj)
            
            # Now check if the input dates are in the holiday set
            if pd.api.types.is_datetime64_any_dtype(date_series):
                # Convert datetime series to string format
                date_strings = date_series.dt.strftime('%Y-%m-%d')
                results = date_strings.isin(holiday_dates_str)
            else:
                # Try to convert to datetime first if it's not already
                try:
                    converted_dates = pd.to_datetime(date_series)
                    date_strings = converted_dates.dt.strftime('%Y-%m-%d')
                    results = date_strings.isin(holiday_dates_str)
                except Exception:
                    # If conversion fails, try direct string comparison
                    if isinstance(date_series, pd.Series) and date_series.dtype == 'object':
                        results = date_series.isin(holiday_dates_str)
            
            # For debugging, print results
            holiday_count = results.sum() if isinstance(results, pd.Series) else sum(results)
            if holiday_count > 0:
                holiday_dates_found = date_series[results].tolist()[:5] if len(date_series[results]) > 0 else []
                # print(f"Found {holiday_count} holidays matching the following dates: {holiday_dates_found}")
        except Exception as e:
            # print(f"Error checking holiday dates: {e}")
            pass
        
        return results
    
    # Ensure we have uk_bank_holidays
    if uk_bank_holidays is None or not isinstance(uk_bank_holidays, pd.DataFrame):
        uk_bank_holidays = pd.DataFrame({'Bank holidays': []})
    
    # If acorn_data is None, create it from daily data
    if acorn_data is None:
        acorn_data = daily.copy()
    
    # Prepare bank holiday dates in multiple formats for flexible matching
    try:
        # Check if we have valid holiday data
        if isinstance(uk_bank_holidays, pd.DataFrame) and len(uk_bank_holidays) > 0:
            holiday_column = None
            
            # Check for 'Bank holidays' column (case insensitive)
            for col in uk_bank_holidays.columns:
                if col.lower() == 'bank holidays':
                    holiday_column = col
                    break
            
            if holiday_column:
                # print(f"Found holiday column: {holiday_column} with {len(uk_bank_holidays)} entries")
                
                # Convert holiday dates to datetime objects
                holiday_dates = pd.to_datetime(uk_bank_holidays[holiday_column], errors='coerce')
                valid_dates = holiday_dates.dropna()
                
                if len(valid_dates) > 0:
                    # Get date objects and datetime objects
                    bank_holiday_dates_list = valid_dates.dt.date.tolist()
                    bank_holiday_dates_datetime = valid_dates.tolist()
                    # Use sets for faster lookups
                    bank_holiday_dates_set = set(bank_holiday_dates_list)
                    # print(f"Processed {len(bank_holiday_dates_set)} valid bank holidays")
                    # print(f"Sample dates: {list(bank_holiday_dates_set)[:3]}")
                else:
                    # print("No valid dates found in holiday column")
                    bank_holiday_dates_list = []
                    bank_holiday_dates_datetime = []
                    bank_holiday_dates_set = set()
            else:
                # print(f"No 'Bank holidays' column found. Available columns: {uk_bank_holidays.columns.tolist()}")
                bank_holiday_dates_list = []
                bank_holiday_dates_datetime = []
                bank_holiday_dates_set = set()
        else:
            # print("No valid bank holiday data provided")
            bank_holiday_dates_list = []
            bank_holiday_dates_datetime = []
            bank_holiday_dates_set = set()
    except Exception as e:
        # print(f"Error processing holiday dates: {e}")
        bank_holiday_dates_list = []
        bank_holiday_dates_datetime = []
        bank_holiday_dates_set = set()
    
    # Handle different possible data structures for acorn_data
    # If acorn_data is a dict of DataFrames or Series
    if isinstance(acorn_data, dict):
        boxplot_dfs = []
        for acorn, data in acorn_data.items():
            if isinstance(data, pd.DataFrame):
                temp_df = data.copy()
            elif isinstance(data, pd.Series):  # Convert Series to DataFrame
                temp_df = series_to_df(data)
            else:  # Handle non-DataFrame, non-Series types
                temp_df = pd.DataFrame(data)
            temp_df['Acorn'] = acorn
            boxplot_dfs.append(temp_df)
        boxplot_df = pd.concat(boxplot_dfs, ignore_index=True) if boxplot_dfs else pd.DataFrame()
    # If acorn_data is already a DataFrame with an Acorn column
    elif isinstance(acorn_data, pd.DataFrame) and 'Acorn' in acorn_data.columns:
        boxplot_df = acorn_data.copy()
    # If we just have the daily DataFrame
    else:
        boxplot_df = daily.copy()
    
    # Prepare data for outlier detection based on data structure
    outlier_points = []
    
    # If acorn_data is a dictionary
    if isinstance(acorn_data, dict):
        for acorn, data in acorn_data.items():
            if isinstance(data, pd.DataFrame):
                temp_df = data.copy()
            else:  # Series
                temp_df = series_to_df(data)
                
            # Find the consumption column - it could be named Conso_kWh or something else
            conso_col = 'Conso_kWh'
            if conso_col not in temp_df.columns and len(temp_df.columns) > 0:
                # Use the first numeric column as the consumption column
                numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    conso_col = numeric_cols[0]
            
            if conso_col in temp_df.columns:
                outlier_idx = get_outlier_indices(temp_df[conso_col])
                if len(outlier_idx) > 0:
                    outlier_df = temp_df.loc[outlier_idx].copy()
                    outlier_df['Acorn'] = acorn
                    if 'Date' in outlier_df.columns:
                        # Use helper function for reliable holiday checking
                        outlier_df['IsHoliday'] = is_holiday(
                            outlier_df['Date'], 
                            bank_holiday_dates_set, 
                            bank_holiday_dates_datetime
                        )
                    else:
                        outlier_df['IsHoliday'] = False
                    # Ensure Conso_kWh exists for plotting
                    if conso_col != 'Conso_kWh':
                        outlier_df['Conso_kWh'] = outlier_df[conso_col]
                    outlier_points.append(outlier_df)
    # If acorn_data is already a DataFrame with Acorn column
    elif isinstance(acorn_data, pd.DataFrame) and 'Acorn' in acorn_data.columns:
        for acorn in acorn_data['Acorn'].unique():
            acorn_subset = acorn_data[acorn_data['Acorn'] == acorn].copy()
            if 'Conso_kWh' in acorn_subset.columns:
                outlier_idx = get_outlier_indices(acorn_subset['Conso_kWh'])
                if len(outlier_idx) > 0:
                    outlier_df = acorn_subset.loc[outlier_idx].copy()
                    if 'Date' in outlier_df.columns:
                        # Use helper function for reliable holiday checking
                        outlier_df['IsHoliday'] = is_holiday(
                            outlier_df['Date'], 
                            bank_holiday_dates_set, 
                            bank_holiday_dates_datetime
                        )
                    else:
                        outlier_df['IsHoliday'] = False
                    outlier_points.append(outlier_df)
    # If we just have the daily DataFrame
    else:
        # Group by Acorn
        for acorn in daily['Acorn'].unique():
            acorn_subset = daily[daily['Acorn'] == acorn].copy()
            outlier_idx = get_outlier_indices(acorn_subset['Conso_kWh'])
            if len(outlier_idx) > 0:
                outlier_df = acorn_subset.loc[outlier_idx].copy()
                if 'Date' in outlier_df.columns:
                    # Use helper function for reliable holiday checking
                    outlier_df['IsHoliday'] = is_holiday(
                        outlier_df['Date'], 
                        bank_holiday_dates_set, 
                        bank_holiday_dates_datetime
                    )
                else:
                    outlier_df['IsHoliday'] = False
                outlier_points.append(outlier_df)
    
    # Combine all outlier points
    outlier_points_df = pd.concat(outlier_points) if outlier_points else pd.DataFrame(columns=['Acorn', 'Conso_kWh', 'IsHoliday'])
    # Ensure boxplot_df has the necessary columns
    if 'Conso_kWh' not in boxplot_df.columns and len(boxplot_df.columns) > 0:
        # Find a suitable numeric column to use
        numeric_cols = boxplot_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Use the first numeric column and rename it
            boxplot_df['Conso_kWh'] = boxplot_df[numeric_cols[0]]
    
    # Boxplot by Acorn
    fig1 = plt.figure(figsize=(8, 6))
    sns.boxplot(x='Acorn', y='Conso_kWh', data=boxplot_df, showfliers=False)
    
    # Get unique Acorn values for positioning
    unique_acorns = boxplot_df['Acorn'].unique()
    acorn_positions = {acorn: idx for idx, acorn in enumerate(unique_acorns)}
    
    # Plot outliers
    for acorn in unique_acorns:
        acorn_outliers = outlier_points_df[outlier_points_df['Acorn'] == acorn]
        if not acorn_outliers.empty:
            x_pos = acorn_positions[acorn]
            
            # Create a color list based on IsHoliday - explicitly convert boolean to avoid issues
            colors = ['red' if is_holiday else 'black' for is_holiday in acorn_outliers['IsHoliday']]
            
            # Print for debugging
            holiday_count = sum(acorn_outliers['IsHoliday'])
            if holiday_count > 0:
                pass
                # print(f"{acorn}: Found {holiday_count} holiday outliers out of {len(acorn_outliers)} total outliers")
            
            plt.scatter(
                np.full(acorn_outliers.shape[0], x_pos),
                acorn_outliers['Conso_kWh'],
                c=colors,
                s=40, zorder=5, label=None
            )
    plt.title("Distribution of Daily Consumption by Acorn Type\n(Red: Bank Holiday Outliers)")
    plt.xlabel("Acorn Type")
    plt.ylabel("Daily Consumption (kWh)")
    plt.grid(axis='y')
    
    # Boxplots by Day of the Week
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Get unique Acorn values
    unique_acorns = boxplot_df['Acorn'].unique()
    
    fig2, axes = plt.subplots(1, len(unique_acorns), figsize=(20, 6), sharey=True)
    # Handle single Acorn case
    if len(unique_acorns) == 1:
        axes = [axes]  # Wrap in list to make it iterable
    
    for ax, acorn in zip(axes, unique_acorns):
        # Filter data for current acorn
        df = boxplot_df[boxplot_df['Acorn'] == acorn].copy()
        
        # Add day of week if Date column exists
        if 'Date' in df.columns:
            # Ensure the Date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    # If conversion fails, display error message
                    # print(f"Error converting Date column to datetime for {acorn}: {e}")
                    ax.text(0.5, 0.5, f"Date column not in datetime format for {acorn}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12)
                    continue
                    
            df['Day'] = df['Date'].dt.dayofweek.map(dict(enumerate(day_names)))
            
            # Create boxplot
            sns.boxplot(x='Day', y='Conso_kWh', data=df, ax=ax, showfliers=False)
            
            # Add outliers with holiday highlighting
            for i, day in enumerate(day_names):
                day_data = df[df['Day'] == day]
                if not day_data.empty:
                    outlier_idx = get_outlier_indices(day_data['Conso_kWh'])
                    if len(outlier_idx) > 0:
                        outlier_df = day_data.loc[outlier_idx].copy()
                        # Use helper function for reliable holiday checking
                        outlier_df['IsHoliday'] = is_holiday(
                            outlier_df['Date'], 
                            bank_holiday_dates_set, 
                            bank_holiday_dates_datetime
                        )
                        x_pos = [i] * len(outlier_df)
                        ax.scatter(
                            x_pos,
                            outlier_df['Conso_kWh'],
                            c=outlier_df['IsHoliday'].map({True: 'red', False: 'black'}),
                            s=40, zorder=5
                        )
        else:
            # If there's no Date column, we can't create this plot properly
            ax.text(0.5, 0.5, f"No Date column in data for {acorn}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'{acorn} - by Day')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Consumption (kWh)')
        ax.grid(axis='y')
    
    
    # Boxplots by Season
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    # Get unique Acorn values
    unique_acorns = boxplot_df['Acorn'].unique()
    
    fig3, axes = plt.subplots(1, len(unique_acorns), figsize=(20, 6), sharey=True)
    # Handle single Acorn case
    if len(unique_acorns) == 1:
        axes = [axes]  # Wrap in list to make it iterable
    
    for ax, acorn in zip(axes, unique_acorns):
        # Filter data for current acorn
        df = boxplot_df[boxplot_df['Acorn'] == acorn].copy()
        
        # Add season if Date column exists
        if 'Date' in df.columns:
            # Ensure the Date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    # If conversion fails, display error message
                    # print(f"Error converting Date column to datetime for {acorn}: {e}")
                    ax.text(0.5, 0.5, f"Date column not in datetime format for {acorn}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12)
                    continue
                    
            df['Season'] = pd.cut(df['Date'].dt.month, bins=[0, 3, 6, 9, 12], labels=seasons, right=False)
            
            # Create boxplot
            sns.boxplot(x='Season', y='Conso_kWh', data=df, ax=ax, showfliers=False)
            
            # Add outliers with holiday highlighting
            for i, season in enumerate(seasons):
                season_data = df[df['Season'] == season]
                if not season_data.empty:
                    outlier_idx = get_outlier_indices(season_data['Conso_kWh'])
                    if len(outlier_idx) > 0:
                        outlier_df = season_data.loc[outlier_idx].copy()
                        # Use helper function for reliable holiday checking
                        outlier_df['IsHoliday'] = is_holiday(
                            outlier_df['Date'], 
                            bank_holiday_dates_set, 
                            bank_holiday_dates_datetime
                        )
                        x_pos = [i] * len(outlier_df)
                        ax.scatter(
                            x_pos,
                            outlier_df['Conso_kWh'],
                            c=outlier_df['IsHoliday'].map({True: 'red', False: 'black'}),
                            s=40, zorder=5
                        )
        else:
            # If there's no Date column, we can't create this plot properly
            ax.text(0.5, 0.5, f"No Date column in data for {acorn}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'{acorn} - by Season')
        ax.set_xlabel('Season')
        ax.set_ylabel('Consumption (kWh)')
        ax.grid(axis='y')
    
    # Boxplots by Month
    month_labels = list(range(1, 13))
    
    # Get unique Acorn values
    unique_acorns = boxplot_df['Acorn'].unique()
    
    fig4, axes = plt.subplots(1, len(unique_acorns), figsize=(20, 6), sharey=True)
    # Handle single Acorn case
    if len(unique_acorns) == 1:
        axes = [axes]  # Wrap in list to make it iterable
    
    for ax, acorn in zip(axes, unique_acorns):
        # Filter data for current acorn
        df = boxplot_df[boxplot_df['Acorn'] == acorn].copy()
        
        # Add month if Date column exists
        if 'Date' in df.columns:
            # Ensure the Date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    # If conversion fails, display error message
                    # print(f"Error converting Date column to datetime for {acorn}: {e}")
                    ax.text(0.5, 0.5, f"Date column not in datetime format for {acorn}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12)
                    continue
                    
            df['Month'] = df['Date'].dt.month
            
            # Create boxplot
            sns.boxplot(x='Month', y='Conso_kWh', data=df, ax=ax, showfliers=False)
            
            # Add outliers with holiday highlighting
            for i, month in enumerate(month_labels):
                month_data = df[df['Month'] == month]
                if not month_data.empty:
                    outlier_idx = get_outlier_indices(month_data['Conso_kWh'])
                    if len(outlier_idx) > 0:
                        outlier_df = month_data.loc[outlier_idx].copy()
                        # Use helper function for reliable holiday checking
                        outlier_df['IsHoliday'] = is_holiday(
                            outlier_df['Date'], 
                            bank_holiday_dates_set, 
                            bank_holiday_dates_datetime
                        )
                        x_pos = [i] * len(outlier_df)
                        ax.scatter(
                            x_pos,
                            outlier_df['Conso_kWh'],
                            c=outlier_df['IsHoliday'].map({True: 'red', False: 'black'}),
                            s=40, zorder=5
                        )
        else:
            # If there's no Date column, we can't create this plot properly
            ax.text(0.5, 0.5, f"No Date column in data for {acorn}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'{acorn} - by Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Consumption (kWh)')
        ax.grid(axis='y')
        
    return fig1, fig2, fig3, fig4

# ############################################################################################################

# Global constants
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall'] 