import pandas as pd

def preprocess_weather_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the weather daily data by converting the columns with dates to datetime format
    and clean the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame containing weather daily data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with 'date' as the index.
    """
    
    datetime_cols = [col for col in df.columns if 'time' in col.lower()]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])

    df.cloudCover = df.cloudCover.interpolate()
    df = df.drop(columns='uvIndexTime')

    return df