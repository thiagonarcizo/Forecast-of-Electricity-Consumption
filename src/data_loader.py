import pandas as pd
from preprocessing import preprocess_weather_daily

def load_weather_data() -> pd.DataFrame:

    """
    Load and preprocess the weather data from a CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with 'date' as the index.
    """
    df = pd.read_csv(r'data\00_raw\weather_daily_darksky.csv')
    df = preprocess_weather_daily(df)

    return df
