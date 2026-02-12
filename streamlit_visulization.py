import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from io import StringIO
import tempfile
import os

# Import your existing modules
import data_loader  # data_loader.py must define load_data(file_path)
import indicators   # indicators.py must define calculate_indicators(df)

# --- File Upload and Data Processing ---

st.sidebar.title("File Upload & Processing")

uploaded_file = st.sidebar.file_uploader("Upload your OHLC CSV file", type=["csv"])
data_status = None  # We'll use this to keep track of whether the file was successfully loaded/processed

if uploaded_file is not None:
    # Save the uploaded file to a temporary file so we can pass a file path to data_loader.load_data()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.flush()
        temp_file_path = tmp_file.name

    st.sidebar.write("Processing file with data_loader.py...")
    df_loaded = data_loader.load_data(temp_file_path)  # This will print messages to console
    # Remove the temporary file after loading
    os.unlink(temp_file_path)

    if df_loaded is None:
        st.error("Error loading data. Please check your file for critical errors.")
        st.stop()
    else:
        st.sidebar.success("Data loaded successfully with no critical errors.")
        st.sidebar.write("Calculating indicators...")
        df_processed = indicators.calculate_indicators(df_loaded)
        st.sidebar.success("Indicators calculated successfully.")
        # Use the processed DataFrame in the dashboard
        df = df_processed.copy()
else:
    st.sidebar.info("Please upload an OHLC CSV file.")
    st.stop()

# --- Date Range Filter ---
# Instead of using the index, we now get the date range from the 'Date' column
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

st.sidebar.title("Chart Controls")

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
if len(date_range) != 2:
    st.sidebar.error("Please select a start and end date.")
    st.stop()
start_date, end_date = date_range
# Filter the data based on the selected date range using the 'Date' column
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# --- Define Indicator Options ---
# Exclude standard OHLC columns
all_indicators = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
range_indicators = [col for col in all_indicators if 'Range' in col]
# Define moving averages & Bollinger bands (and similar) by common keywords
ma_indicators = [col for col in df.columns if ('SMA' in col or 'EMA' in col or 'BB' in col or 'Low_prev' in col or 'High_prev' in col)]
other_indicators = [col for col in all_indicators if col not in ma_indicators]
range_ratio_indicators = [col for col in range_indicators if '/' in col]  # Assuming range ratios contain '/'

# --- Sidebar: NaN Handling Option ---
nan_option = st.sidebar.radio(
    "How do you want to handle rows with NaN values for plotting?",
    options=["Drop rows with NaN values", "Keep rows with NaN values"],
    index=0
)
drop_nan = True if nan_option == "Drop rows with NaN values" else False

# --- More Sidebar Controls ---
chart_type = st.sidebar.selectbox("Select Chart Type", options=["OHLC (Candlestick)", "Line (Closing Price)"])
selected_mas = st.sidebar.multiselect("Select up to 15 Moving Averages / Bollinger Bands", options=ma_indicators, max_selections=15)

subchart1_indicator = st.sidebar.selectbox("Select Indicator for Subchart 1", options=other_indicators, index=0 if other_indicators else 0)
if len(other_indicators) > 1:
    subchart2_indicator = st.sidebar.selectbox("Select Indicator for Subchart 2", options=other_indicators, index=1)
else:
    subchart2_indicator = None

selected_percentile_inds = st.sidebar.multiselect("Select Indicators for Percentile Table", options=all_indicators)
selected_stats_inds = st.sidebar.multiselect("Select Indicators for Statistical Analysis", options=all_indicators)
selected_corr_inds = st.sidebar.multiselect("Select Indicators for Correlation Matrix", options=all_indicators, default=range_ratio_indicators)

# --- Download / Export Button ---
def convert_df_to_csv(dataframe):
    return dataframe.to_csv().encode('utf-8')

csv_data = convert_df_to_csv(df_filtered)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv_data,
    file_name='filtered_data.csv',
    mime='text/csv'
)

# --- Functions to Build Figures ---
def build_main_chart(chart_type, selected_mas, data, drop_nan):
    # Either drop rows with NaN or keep complete data based on the user's choice
    if drop_nan:
        data = data.dropna().reset_index(drop=True)
    else:
        data = data.copy()
    fig = go.Figure()
    
    if chart_type == "OHLC (Candlestick)":
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name='Close Price'
        ))
    
    # Add selected moving averages / Bollinger Bands
    for ma in selected_mas:
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data[ma],
            mode='lines',
            name=ma
        ))
    
    fig.update_layout(
        title="Main Chart",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x unified'
    )
    return fig

def build_subchart(indicator, title, data, drop_nan):
    # Either drop rows with NaN or keep complete data based on the user's choice
    if drop_nan:
        data = data.dropna().reset_index(drop=True)
    else:
        data = data.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data[indicator],
        mode='lines',
        name=indicator
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=indicator,
        height=300,  # Subcharts are roughly 1/4 the height of the main chart
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x unified'
    )
    return fig

def build_percentile_table(selected_indicators, data):
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentile_data = [{"Percentile": p} for p in percentiles]
    for col in selected_indicators:
        for i, p in enumerate(percentiles):
            percentile_data[i][col] = data[col].quantile(p / 100)
    return pd.DataFrame(percentile_data)

def build_stats_table(selected_indicators, data):
    stats_df = data[selected_indicators].describe().reset_index().rename(columns={'index': 'Statistic'})
    return stats_df

def build_corr_table(selected_indicators, data):
    corr_df = data[selected_indicators].corr().reset_index()
    return corr_df

def build_corr_heatmap(selected_indicators, data):
    corr = data[selected_indicators].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title="Correlation Heatmap",
        height=500,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# --- Render the App Layout ---
st.title("Backtesting Visualization Dashboard")

st.markdown("### Main Chart")
main_chart_fig = build_main_chart(chart_type, selected_mas, df_filtered, drop_nan)
st.plotly_chart(main_chart_fig, use_container_width=True)

st.markdown("### Subchart 1")
subchart1_fig = build_subchart(subchart1_indicator, f"Subchart 1: {subchart1_indicator}", df_filtered, drop_nan)
st.plotly_chart(subchart1_fig, use_container_width=True)

if subchart2_indicator:
    st.markdown("### Subchart 2")
    subchart2_fig = build_subchart(subchart2_indicator, f"Subchart 2: {subchart2_indicator}", df_filtered, drop_nan)
    st.plotly_chart(subchart2_fig, use_container_width=True)
else:
    st.write("Subchart 2 not available.")

st.markdown("### Percentile Table")
if selected_percentile_inds:
    percentile_df = build_percentile_table(selected_percentile_inds, df_filtered)
    st.dataframe(percentile_df)
else:
    st.write("Select indicators from the sidebar to display the percentile table.")

st.markdown("### Statistical Analysis")
if selected_stats_inds:
    stats_df = build_stats_table(selected_stats_inds, df_filtered)
    st.dataframe(stats_df)
else:
    st.write("Select indicators from the sidebar to display the statistical analysis table.")

st.markdown("### Correlation Matrix")
if selected_corr_inds:
    corr_df = build_corr_table(selected_corr_inds, df_filtered)
    st.dataframe(corr_df)
else:
    st.write("Select indicators from the sidebar to display the correlation matrix.")

st.markdown("### Correlation Heatmap")
if selected_corr_inds:
    corr_heatmap_fig = build_corr_heatmap(selected_corr_inds, df_filtered)
    st.plotly_chart(corr_heatmap_fig, use_container_width=True)
else:
    st.write("Select indicators from the sidebar to display the correlation heatmap.")