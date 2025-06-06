import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Interactive Prediction", layout="wide")

st.title("💡 Interactive Electricity Consumption Prediction")
st.markdown("""
This page allows you to interact with the electricity consumption predictions and compare them with historical data.
Use the filters on the sidebar to explore the data for different ACORN groups and date ranges.
""")


@st.cache_data
def load_data(actual_path, predicted_path):
    """Loads and merges actual and predicted consumption data."""
    if not os.path.exists(actual_path) or not os.path.exists(predicted_path):
        st.error("Data file not found.")
        return pd.DataFrame()

    df_actual = pd.read_csv(actual_path)
    df_predict = pd.read_csv(predicted_path)

    df_actual['Date'] = pd.to_datetime(df_actual['Date'])
    df_predict['Date'] = pd.to_datetime(df_predict['Date'])

    df_actual.rename(columns={'Conso_kWh': 'Actual'}, inplace=True)
    df_predict.rename(columns={'Conso_kWh_predict': 'Predicted'}, inplace=True)

    df_merged = pd.merge(
        df_actual[['Date', 'Acorn', 'Actual']],
        df_predict[['Date', 'Acorn', 'Predicted']],
        on=['Date', 'Acorn'],
        how='outer'
    )
    
    df_merged.sort_values(by=['Acorn', 'Date'], inplace=True)
    return df_merged

actual_data_path = 'data/02_processed/csv/group_4_daily.csv'
predicted_data_path = 'data/02_processed/csv/group_4_daily_predict.csv'
df = load_data(actual_data_path, predicted_data_path)

if df.empty:
    st.stop()

st.sidebar.header("Filter Options")

unique_acorns = df['Acorn'].unique()
acorn_groups = st.sidebar.multiselect(
    "Select ACORN Group(s):",
    options=unique_acorns,
    default=unique_acorns
)

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key='date_range_selector'
)

df_selection = df[
    (df['Acorn'].isin(acorn_groups)) &
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
].copy()


st.header("📊 Consumption Dashboard")

if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
else:
    df_selection['Consumption'] = df_selection['Actual'].fillna(df_selection['Predicted'])
    total_consumption = df_selection['Consumption'].sum()
    average_consumption = df_selection['Consumption'].mean()
    peak_consumption_date = df_selection.loc[df_selection['Consumption'].idxmax()]['Date'].date()
    peak_consumption_value = df_selection['Consumption'].max()

    st.subheader("Key Metrics for Selected Period")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Consumption (kWh)", value=f"{total_consumption:,.0f}")
    with col2:
        st.metric(label="Avg. Daily Consumption (kWh)", value=f"{average_consumption:,.2f}")
    with col3:
        st.metric(label=f"Peak Consumption ({peak_consumption_date})", value=f"{peak_consumption_value:,.2f} kWh")

    st.markdown("---")

    st.subheader("Consumption Trend: Actual vs. Predicted")
    
    df_plot = df_selection.melt(
        id_vars=['Date', 'Acorn'],
        value_vars=['Actual', 'Predicted'],
        var_name='Source',
        value_name='Consumption'
    )
    df_plot.dropna(subset=['Consumption'], inplace=True)

    fig = px.line(
        df_plot,
        x='Date',
        y='Consumption',
        color='Source',
        line_dash='Acorn',
        title='Daily Electricity Consumption: Actual vs. Predicted',
        labels={'Consumption': 'Consumption (kWh)', 'Date': 'Date', 'Source': 'Data Source', 'Acorn': 'ACORN Group'},
        hover_data={'Date': '|%B %d, %Y', 'Consumption': ':.2f'}
    )
    
    first_prediction_date = df_selection[df_selection['Predicted'].notna()]['Date'].min()
    if pd.notna(first_prediction_date) and first_prediction_date >= df_selection['Date'].min() and first_prediction_date <= df_selection['Date'].max():
        fig.add_vline(x=first_prediction_date, line_width=2, line_dash="dash", line_color="green", annotation_text="Prediction Start")

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)),
        legend_title_text='Data Source'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consumption Breakdown by ACORN Group")
    
    col1, col2 = st.columns([2,1])
    with col1:
        acorn_total = df_selection.groupby('Acorn')['Consumption'].sum().sort_values(ascending=False)
        fig_bar = px.bar(
            acorn_total,
            x=acorn_total.index,
            y=acorn_total.values,
            labels={'x': 'ACORN Group', 'y': 'Total Consumption (kWh)'},
            title="Total Consumption per ACORN Group"
        )
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.dataframe(acorn_total.reset_index().rename(columns={'index': 'Acorn', 'Consumption': 'Total kWh'}))

    with st.expander("View Raw Data for Selection"):
        st.dataframe(df_selection.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Consumption': '{:.2f}'}), use_container_width=True)

