import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------
# CACHING (FAST LOAD)
# ---------------------------------------------------
@st.cache_data
def load_campaign():
    return pd.read_csv('data/campaign_performance.csv', parse_dates=['date'])

@st.cache_data
def load_customers():
    return pd.read_csv("data/customer_data.csv")

@st.cache_data
def load_attribution():
    return pd.read_csv("data/channel_attribution.csv")

@st.cache_data
def load_correlation():
    return pd.read_csv("data/correlation_matrix.csv", index_col=0)


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="NovaMart Dashboard", layout="wide")

page = st.sidebar.radio(
    "Navigation",
    ["Executive Summary", "Campaign Performance", "Customer Analytics"]
)

# ---------------------------------------------------
# FORMAT HELPERS
# ---------------------------------------------------
def fmt_money(x):
    return f"₹{x:,.0f}"

def fmt_num(x):
    return f"{x:,.0f}"

# ===============================================================
# 1️⃣ EXECUTIVE SUMMARY
# ===============================================================
if page == "Executive Summary":

    df = load_campaign()
    st.title("🛒 NovaMart – Executive Summary")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", fmt_money(df['revenue'].sum()))
    col2.metric("Ad Spend", fmt_money(df['spend'].sum()))
    col3.metric("Conversions", fmt_num(df['conversions'].sum()))

    st.markdown("---")

    # Revenue Trend
    st.subheader("📈 Revenue Trend (7-day MA)")
    df_daily = df.groupby("date")["revenue"].sum().rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily.index,
        y=df_daily.values,
        mode="lines",
        line=dict(width=2)
    ))
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Channel Pie
    st.subheader("📊 Revenue by Channel")
    ch = df.groupby("channel")["revenue"].sum()

    fig = go.Figure(go.Pie(labels=ch.index, values=ch.values, hole=0.45))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ===============================================================
# 2️⃣ CAMPAIGN PERFORMANCE
# ===============================================================
elif page == "Campaign Performance":

    df = load_campaign()
    st.title("📈 Campaign Performance")

    # Filters
    channels = ["All"] + sorted(df["channel"].unique())
    regions = ["All"] + sorted(df["region"].unique())

    c1, c2 = st.columns(2)
    f_channel = c1.selectbox("Channel", channels)
    f_region = c2.selectbox("Region", regions)

    filt = df.copy()
    if f_channel != "All":
        filt = filt[filt["channel"] == f_channel]
    if f_region != "All":
        filt = filt[filt["region"] == f_region]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Clicks", fmt_num(filt["clicks"].sum()))
    col2.metric("Conversions", fmt_num(filt["conversions"].sum()))
    col3.metric("Revenue", fmt_money(filt["revenue"].sum()))

    # Revenue by Channel
    st.subheader("🔗 Revenue by Channel")
    grp = filt.groupby("channel")["revenue"].sum()

    fig = go.Figure(go.Bar(
        x=grp.values,
        y=grp.index,
        orientation="h"
    ))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("🗺️ ROAS Heatmap")

    filt = filt.copy()
    filt["roas"] = filt["revenue"] / filt["spend"].replace(0, 1)
    pivot = filt.pivot_table(
        index="region", columns="channel", values="roas", aggfunc="mean"
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn"
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# ===============================================================
# 3️⃣ CUSTOMER ANALYTICS
# ===============================================================
elif page == "Customer Analytics":

    cust = load_customers()
    corr = load_correlation()

    st.title("👥 Customer Analytics")

    churn_rate = cust["churn_probability"].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", fmt_num(len(cust)))
    col2.metric("Avg LTV", fmt_money(cust["lifetime_value"].mean()))
    col3.metric("Churn Probability", f"{churn_rate:.1f}%")

    st.markdown("---")

    # Segments Pie
    st.subheader("📊 Customer Segments")
    seg = cust["customer_segment"].value_counts()

    fig = go.Figure(go.Pie(labels=seg.index, values=seg.values, hole=0.45))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📌 Customer Feature Correlation")

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu"
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
