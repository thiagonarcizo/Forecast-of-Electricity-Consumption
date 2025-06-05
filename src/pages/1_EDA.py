import streamlit as st
from src.streamlit_helpers import (
    load_all_data,
    create_boxplot_by_acorn,
    create_heatmap_by_acorn,
    create_load_duration_curves,
    create_temporal_boxplots,
    create_seasonal_analysis,
    print_seasonal_summary,
    plot_daily_acorn_consumption,
    plot_daily_acorn_outlier_boxplots,
    add_temporal_features
)
import pandas as pd

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
acorn_selection = st.selectbox("Select Acorn Group to Analyze", acorn_groups)

# Filter data for selected Acorn group
daily_data_acorn = daily_data[daily_data['Acorn'] == acorn_selection]
half_hourly_data_acorn = half_hourly_data[half_hourly_data['Acorn'] == acorn_selection]

# Display plots
st.header(f"Analysis for Acorn Group: {acorn_selection}")

st.subheader("Boxplot of Consumption by Acorn Group")
fig = create_boxplot_by_acorn(half_hourly_data_acorn)
st.pyplot(fig)

st.subheader("Consumption Heatmap")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fig = create_heatmap_by_acorn(half_hourly_data_acorn, [acorn_selection], day_order)
st.pyplot(fig)

st.subheader("Load Duration Curve")
fig = create_load_duration_curves(half_hourly_data_acorn, [acorn_selection])
st.pyplot(fig)

st.subheader("Temporal Boxplots")
fig = create_temporal_boxplots(half_hourly_data_acorn, [acorn_selection], day_order)
st.pyplot(fig)

st.subheader("Seasonal Analysis")
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
fig = create_seasonal_analysis(half_hourly_data_acorn, [acorn_selection], season_order)
st.pyplot(fig)

st.subheader("Seasonal Summary")
summary = print_seasonal_summary(half_hourly_data_acorn, [acorn_selection], season_order)
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