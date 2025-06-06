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
    if not os.path.exists(actual_path):
        st.error(f"Data file not found: {actual_path}")
        return pd.DataFrame()
    if not os.path.exists(predicted_path):
        st.error(f"Data file not found: {predicted_path}")
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

st.sidebar.header("Filter Options")

model_choice = st.sidebar.selectbox(
    "Choose a prediction model:",
    ('LSTM', 'LightGBM', 'MLP', 'SVM', 'SARIMAX'),
    key="model_selector"
)

actual_data_path = 'data/02_processed/csv/group_4_daily.csv'
if model_choice == 'LightGBM':
    predicted_data_path = 'data/02_processed/csv/group_4_daily_predict_lightgbm.csv'
elif model_choice == 'LSTM':
    predicted_data_path = 'data/02_processed/csv/group_4_daily_predict_lstmRF.csv'
elif model_choice == 'MLP':
    predicted_data_path = 'data/02_processed/csv/group_4_daily_predict_mlp.csv'
elif model_choice == 'SVM':
    predicted_data_path = 'data/02_processed/csv/group_4_daily_predict_svm.csv'
elif model_choice == 'SARIMAX':
    predicted_data_path = 'data/02_processed/csv/group_4_daily_predict_sarimax.csv'
else:
    st.error("Invalid model choice. Please select a valid model.")
    st.stop()

df = load_data(actual_data_path, predicted_data_path)

if df.empty:
    st.stop()

unique_acorns = df['Acorn'].unique()
acorn_groups = st.sidebar.multiselect(
    "Select ACORN Group(s):",
    options=unique_acorns,
    default=unique_acorns
)

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

if st.sidebar.button("Select All Dates"):
    st.session_state.date_range_selector = (min_date, max_date)
    st.rerun()

dates = st.sidebar.date_input(
    "Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key='date_range_selector'
)

if isinstance(dates, tuple):
    if len(dates) == 2:
        start_date, end_date = dates
    else:
        start_date = end_date = dates[0]
else:
    start_date = end_date = dates

df_selection = df[
    (df['Acorn'].isin(acorn_groups)) &
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
].copy()


st.header("📊 Consumption Dashboard")

if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
else:
    df_selection['Conso_kWh'] = df_selection['Actual'].fillna(df_selection['Predicted'])
    total_consumption = df_selection['Conso_kWh'].sum()
    average_consumption = df_selection['Conso_kWh'].mean()
    peak_consumption_date = df_selection.loc[df_selection['Conso_kWh'].idxmax()]['Date'].date()
    peak_consumption_value = df_selection['Conso_kWh'].max()

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
        fig.add_shape(
            type="line",
            x0=first_prediction_date,
            y0=0,
            x1=first_prediction_date,
            y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=first_prediction_date,
            y=1.05,
            yref="paper",
            text="Prediction Start",
            showarrow=False,
            yshift=10,
            font=dict(color="green")
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)),
        legend_title_text='Data Source'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header(f"📈 Focus on {model_choice} Prediction Period")

    df_predicted_period = df_selection[df_selection['Predicted'].notna()].copy()

    if df_predicted_period.empty:
        st.warning("No prediction data available for the selected date range.")
    else:
        st.subheader("Key Metrics for Prediction Period")
        
        predicted_total = df_predicted_period['Predicted'].sum()
        predicted_average = df_predicted_period['Predicted'].mean()
        peak_predicted_date = df_predicted_period.loc[df_predicted_period['Predicted'].idxmax()]['Date'].date()
        peak_predicted_value = df_predicted_period['Predicted'].max()

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.metric(label="Total Predicted Consumption (kWh)", value=f"{predicted_total:,.0f}")
        with p_col2:
            st.metric(label="Avg. Daily Predicted Consumption (kWh)", value=f"{predicted_average:,.2f}")
        with p_col3:
            st.metric(label=f"Peak Prediction ({peak_predicted_date})", value=f"{peak_predicted_value:,.2f} kWh")
        
        st.subheader("Predicted Consumption Trend")
        fig_pred_only = px.line(
            df_predicted_period,
            x='Date',
            y='Predicted',
            color='Acorn',
            title=f'{model_choice} Predicted Daily Consumption',
            labels={'Predicted': 'Predicted Consumption (kWh)', 'Date': 'Date', 'Acorn': 'ACORN Group'},
            hover_data={'Date': '|%B %d, %Y', 'Predicted': ':.2f'}
        )
        fig_pred_only.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False)),
            legend_title_text='ACORN Groups'
        )
        st.plotly_chart(fig_pred_only, use_container_width=True)

    st.subheader("Consumption Breakdown by ACORN Group")
    
    col1, col2 = st.columns([2,1])
    with col1:
        acorn_total = df_selection.groupby('Acorn')['Conso_kWh'].sum().sort_values(ascending=False)
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
        st.dataframe(acorn_total.reset_index().rename(columns={'index': 'Acorn', 'Conso_kWh': 'Total kWh'}))

    with st.expander("View Raw Data for Selection"):
        st.dataframe(df_selection.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Conso_kWh': '{:.2f}'}), use_container_width=True)

