import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Home Power Cost Predictor", layout="wide")
st.title("Home Power Cost Predictor")
st.caption("Predicts daily electricity cost based on temperature and date.")


@st.cache_resource
def train_model():
    df = pd.read_csv("data/dailyCost4_18_2023_to_4_17_2026.csv", encoding="utf-8-sig")
    df = df[pd.to_datetime(df["Usage date"], format="%m/%d/%Y", errors="coerce").notna()].copy()
    df[["Meter read date", "Usage date"]] = df[["Meter read date", "Usage date"]].apply(
        pd.to_datetime, format="%m/%d/%Y"
    )
    df["Total cost"] = pd.to_numeric(
        df["Total cost"].astype(str).str.replace("$", "", regex=False).str.strip('"'),
        errors="coerce",
    )
    df = df.dropna(subset=["Total cost", "High temperature (F)", "Low temperature (F)"])
    df = df.sort_values("Usage date").reset_index(drop=True)

    df["avg_temp"]    = (df["High temperature (F)"] + df["Low temperature (F)"]) / 2
    df["temp_range"]  = df["High temperature (F)"] - df["Low temperature (F)"]
    df["month"]       = df["Usage date"].dt.month
    df["day_of_week"] = df["Usage date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = df["month"].map({
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
    })

    features = ["High temperature (F)", "Low temperature (F)", "avg_temp",
                "temp_range", "month", "day_of_week", "is_weekend", "season"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(df[features], df["Total cost"])
    return model, df, features


model, df, FEATURES = train_model()


def make_features(date, high, low):
    return pd.DataFrame([{
        "High temperature (F)": high,
        "Low temperature (F)":  low,
        "avg_temp":             (high + low) / 2,
        "temp_range":           high - low,
        "month":                date.month,
        "day_of_week":          date.weekday(),
        "is_weekend":           int(date.weekday() >= 5),
        "season":               {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[date.month],
    }])


tab1, tab2 = st.tabs(["Single Day", "Multi-Day Forecast"])

with tab1:
    st.subheader("Predict a Single Day")
    col1, col2, col3 = st.columns(3)
    with col1:
        single_date = st.date_input("Date", value=pd.Timestamp.today())
    with col2:
        high_temp = st.number_input("High Temp (°F)", min_value=30, max_value=130, value=90)
    with col3:
        low_temp = st.number_input("Low Temp (°F)", min_value=20, max_value=110, value=65)

    if st.button("Predict", key="single"):
        if low_temp >= high_temp:
            st.error("Low temp must be less than high temp.")
        else:
            pred = model.predict(make_features(single_date, high_temp, low_temp))[0]
            st.metric("Predicted Cost", f"${pred:.2f}")

with tab2:
    st.subheader("Multi-Day Forecast")
    st.info("Enter up to 7 days of temperature forecasts.")

    num_days = st.slider("Number of days", 2, 7, 5)
    rows = []
    for i in range(num_days):
        date = pd.Timestamp.today().date() + pd.Timedelta(days=i)
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            d = st.date_input("Date", value=date, key=f"date_{i}")
        with c2:
            h = st.number_input("High °F", min_value=30, max_value=130, value=90, key=f"high_{i}")
        with c3:
            l = st.number_input("Low °F", min_value=20, max_value=110, value=65, key=f"low_{i}")
        rows.append((d, h, l))

    if st.button("Forecast", key="multi"):
        results = []
        for d, h, l in rows:
            pred = model.predict(make_features(d, h, l))[0]
            results.append({"Date": d, "High °F": h, "Low °F": l, "Predicted Cost ($)": round(pred, 2)})

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results_df["Date"].astype(str),
            y=results_df["Predicted Cost ($)"],
            marker_color="crimson",
            text=results_df["Predicted Cost ($)"].apply(lambda x: f"${x:.2f}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Forecasted Daily Cost",
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Historical Data")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df["Usage date"], y=df["Total cost"],
    mode="lines", name="Actual Cost",
    line=dict(color="steelblue", width=1.5)
))
fig2.update_layout(
    title="Daily Electricity Cost (Historical)",
    xaxis_title="Date", yaxis_title="Cost ($)",
    template="plotly_white", hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True)
