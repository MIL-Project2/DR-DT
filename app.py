import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="HVAC Backtest Dashboard", layout="wide")
st.title("🏢 Occupancy-Driven HVAC Backtesting")

# Load results
@st.cache_data
def load_data():
    df = pd.read_csv('backtest_results.csv', parse_dates=['timestamp'], index_col='timestamp')
    # Compute cumulative energy (kWh)
    df['energy_pred'] = df['Q_hvac_pred'].cumsum() / 3600e3
    df['energy_true'] = df['Q_hvac_true'].cumsum() / 3600e3
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    [df.index.min().date(), df.index.max().date()]
)
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_filtered = df.loc[mask]
else:
    df_filtered = df

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    rmse_temp = np.sqrt(np.mean((df_filtered['T_pred'] - df_filtered['T_true'])**2))
    st.metric("Temperature RMSE", f"{rmse_temp:.2f} °C")
with col2:
    energy_diff = df_filtered['energy_pred'].iloc[-1] - df_filtered['energy_true'].iloc[-1]
    st.metric("Energy Difference (Pred - True)", f"{energy_diff:.2f} kWh")
with col3:
    comfort_true = ((df_filtered['T_true'] < 20) | (df_filtered['T_true'] > 26)).mean()
    comfort_pred = ((df_filtered['T_pred'] < 20) | (df_filtered['T_pred'] > 26)).mean()
    st.metric("Comfort Violation (True)", f"{comfort_true*100:.1f}%")
    st.metric("Comfort Violation (Pred)", f"{comfort_pred*100:.1f}%")

# Occupancy plot
fig_occ = go.Figure()
fig_occ.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['occ_true'], 
                             name='Actual Occupancy', line=dict(color='blue')))
fig_occ.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['occ_pred'], 
                             name='Predicted Occupancy', line=dict(color='red', dash='dash')))
fig_occ.update_layout(title="Occupancy Over Time", xaxis_title="Time", yaxis_title="Occupancy")

# Temperature plot
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['T_true'], 
                              name='Actual Temp', line=dict(color='green')))
fig_temp.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['T_pred'], 
                              name='Predicted Temp', line=dict(color='orange', dash='dash')))
fig_temp.add_hline(y=20, line_dash="dot", line_color="gray", annotation_text="Comfort lower")
fig_temp.add_hline(y=26, line_dash="dot", line_color="gray", annotation_text="Comfort upper")
fig_temp.update_layout(title="Indoor Temperature", xaxis_title="Time", yaxis_title="Temperature (°C)")

# Energy plot
fig_energy = go.Figure()
fig_energy.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['energy_true'], 
                                name='Actual Energy', line=dict(color='purple')))
fig_energy.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['energy_pred'], 
                                name='Predicted Energy', line=dict(color='brown', dash='dash')))
fig_energy.update_layout(title="Cumulative HVAC Energy", xaxis_title="Time", yaxis_title="Energy (kWh)")

# Show plots
st.plotly_chart(fig_occ, use_container_width=True)
st.plotly_chart(fig_temp, use_container_width=True)
st.plotly_chart(fig_energy, use_container_width=True)

# Optional: scatter of predicted vs actual energy
st.subheader("Energy Scatter")
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=df_filtered['energy_true'], y=df_filtered['energy_pred'], 
                                 mode='markers', marker=dict(color='darkblue')))
fig_scatter.add_trace(go.Scatter(x=[df_filtered['energy_true'].min(), df_filtered['energy_true'].max()],
                                 y=[df_filtered['energy_true'].min(), df_filtered['energy_true'].max()],
                                 mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
fig_scatter.update_layout(title="Predicted vs Actual Cumulative Energy", 
                          xaxis_title="Actual Energy (kWh)", yaxis_title="Predicted Energy (kWh)")
st.plotly_chart(fig_scatter, use_container_width=True)
