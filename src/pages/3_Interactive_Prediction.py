import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Interactive Prediction", layout="wide")

st.title("💡 Interactive Electricity Consumption Prediction")
st.markdown("""
This page allows you to interact with the electricity consumption predictions.
Use the filters on the sidebar to explore the data for different ACORN groups and date ranges.
""")


@st.cache_data
def load_data(file_path):
    """Loads the prediction data from a CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Data file not found at: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

data_path = 'data/02_processed/csv/group_4_daily_predict.csv'
df = load_data(data_path)

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


st.header("📊 Predictions Dashboard")

if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
else:
    total_consumption = df_selection['Conso_kWh_predict'].sum()
    average_consumption = df_selection['Conso_kWh_predict'].mean()
    peak_consumption_date = df_selection.loc[df_selection['Conso_kWh_predict'].idxmax()]['Date'].date()
    peak_consumption_value = df_selection['Conso_kWh_predict'].max()

    st.subheader("Key Metrics for Selected Period")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Predicted Consumption (kWh)", value=f"{total_consumption:,.0f}")
    with col2:
        st.metric(label="Avg. Daily Consumption (kWh)", value=f"{average_consumption:,.2f}")
    with col3:
        st.metric(label=f"Peak Consumption ({peak_consumption_date})", value=f"{peak_consumption_value:,.2f} kWh")

    st.markdown("---")

    st.subheader("Consumption Trend")
    fig = px.line(
        df_selection,
        x='Date',
        y='Conso_kWh_predict',
        color='Acorn',
        title='Predicted Daily Electricity Consumption',
        labels={'Conso_kWh_predict': 'Predicted Consumption (kWh)', 'Date': 'Date', 'Acorn': 'ACORN Group'},
        hover_data={'Date': '|%B %d, %Y', 'Conso_kWh_predict': ':.2f'}
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)),
        legend_title_text='ACORN Groups'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consumption Breakdown by ACORN Group")
    
    col1, col2 = st.columns([2,1])
    with col1:
        acorn_total = df_selection.groupby('Acorn')['Conso_kWh_predict'].sum().sort_values(ascending=False)
        fig_bar = px.bar(
            acorn_total,
            x=acorn_total.index,
            y=acorn_total.values,
            labels={'x': 'ACORN Group', 'y': 'Total Predicted Consumption (kWh)'},
            title="Total Consumption per ACORN Group"
        )
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.dataframe(acorn_total.reset_index().rename(columns={'index': 'Acorn', 'Conso_kWh_predict': 'Total kWh'}))

    with st.expander("View Raw Data for Selection"):
        st.dataframe(df_selection.style.format({'Conso_kWh_predict': '{:.2f}'}), use_container_width=True)

