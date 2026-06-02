import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# ---- page config ----

st.set_page_config(
    page_title="Customer Lifetime Value Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- data generation ----
# simulates ~18 months of ecommerce transactions across 200 customers
# structure mirrors what you'd pull from a real orders table

@st.cache_data
def generate_data():
    np.random.seed(42)

    n_customers = 200
    segments = ["Champions", "Loyal", "At Risk", "New", "Lost"]
    seg_weights = [0.15, 0.25, 0.20, 0.25, 0.15]
    channels = ["Organic", "Paid Search", "Email", "Social", "Direct"]

    customers = pd.DataFrame({
        "customer_id": [f"C{str(i).zfill(4)}" for i in range(1, n_customers + 1)],
        "segment":     np.random.choice(segments, n_customers, p=seg_weights),
        "channel":     np.random.choice(channels, n_customers),
        "join_date":   [
            datetime(2023, 1, 1) + timedelta(days=int(d))
            for d in np.random.randint(0, 365, n_customers)
        ],
        "age_group": np.random.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n_customers),
    })

    # segment-specific purchase behavior
    seg_params = {
        "Champions": {"orders": (8, 14),  "aov": (180, 60),  "gap": (20, 8)},
        "Loyal":     {"orders": (5, 8),   "aov": (120, 40),  "gap": (35, 12)},
        "At Risk":   {"orders": (3, 5),   "aov": (90,  35),  "gap": (60, 20)},
        "New":       {"orders": (1, 3),   "aov": (70,  30),  "gap": (50, 15)},
        "Lost":      {"orders": (1, 2),   "aov": (55,  25),  "gap": (90, 30)},
    }

    records = []
    for _, cust in customers.iterrows():
        params = seg_params[cust["segment"]]
        n_orders = int(np.random.uniform(*params["orders"]))
        for _ in range(n_orders):
            order_date = cust["join_date"] + timedelta(
                days=int(np.random.exponential(params["gap"][0]))
            )
            if order_date > datetime(2024, 6, 30):
                continue
            aov = max(10, np.random.normal(*params["aov"]))
            records.append({
                "customer_id": cust["customer_id"],
                "order_date":  order_date,
                "revenue":     round(aov, 2),
                "segment":     cust["segment"],
                "channel":     cust["channel"],
                "age_group":   cust["age_group"],
            })

    txns = pd.DataFrame(records)
    txns["order_date"] = pd.to_datetime(txns["order_date"])
    txns["month"] = txns["order_date"].dt.to_period("M").dt.to_timestamp()
    return customers, txns


@st.cache_data
def compute_clv(txns, snapshot_date):
    """
    historical CLV = total revenue a customer has generated up to the snapshot date.

    also computes:
      - frequency:    number of orders
      - recency:      days since last order
      - avg_order_value: mean spend per order
      - purchase_rate: orders per month of customer lifetime
      - clv_segment:  percentile-based tier (top 20% = High, next 30% = Mid, rest = Low)

    this is a backward-looking model — it measures realized value rather than
    predicting future spend. for predictive CLV you'd want something like BG/NBD
    but historical CLV is more transparent and easier to explain to stakeholders.
    """
    snap = pd.Timestamp(snapshot_date)
    filtered = txns[txns["order_date"] <= snap]

    clv = filtered.groupby("customer_id").agg(
        total_revenue   = ("revenue",    "sum"),
        order_count     = ("revenue",    "count"),
        avg_order_value = ("revenue",    "mean"),
        first_order     = ("order_date", "min"),
        last_order      = ("order_date", "max"),
        segment         = ("segment",    "first"),
        channel         = ("channel",    "first"),
        age_group       = ("age_group",  "first"),
    ).reset_index()

    clv["recency_days"]    = (snap - clv["last_order"]).dt.days
    clv["tenure_days"]     = (snap - clv["first_order"]).dt.days.clip(lower=1)
    clv["purchase_rate"]   = (clv["order_count"] / clv["tenure_days"] * 30).round(3)

    p80 = clv["total_revenue"].quantile(0.80)
    p50 = clv["total_revenue"].quantile(0.50)
    clv["clv_tier"] = pd.cut(
        clv["total_revenue"],
        bins=[-1, p50, p80, float("inf")],
        labels=["Low", "Mid", "High"]
    )

    return clv.round({"total_revenue": 2, "avg_order_value": 2, "purchase_rate": 3})


# ---- load ----
customers, txns = generate_data()

# ---- sidebar ----

st.sidebar.title("Filters")

date_min = txns["order_date"].min().date()
date_max = txns["order_date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

snapshot = end_date

all_segments = sorted(txns["segment"].unique())
selected_segments = st.sidebar.multiselect("Segments", all_segments, default=all_segments)

all_channels = sorted(txns["channel"].unique())
selected_channels = st.sidebar.multiselect("Channels", all_channels, default=all_channels)

all_ages = sorted(txns["age_group"].unique())
selected_ages = st.sidebar.multiselect("Age groups", all_ages, default=all_ages)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How CLV is calculated**

Historical CLV = sum of all revenue from a customer up to the snapshot date.

Tiers:
- **High** — top 20% by revenue
- **Mid** — next 30%
- **Low** — bottom 50%

This is a backward-looking model. It measures realized value rather than predicting future spend.
""")

# ---- filter data ----

mask = (
    txns["order_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date)) &
    txns["segment"].isin(selected_segments) &
    txns["channel"].isin(selected_channels) &
    txns["age_group"].isin(selected_ages)
)
filtered_txns = txns[mask]

clv_df = compute_clv(filtered_txns, snapshot)

# ---- header ----

st.title("Customer Lifetime Value Dashboard")
st.caption(f"Snapshot date: {snapshot}  ·  {len(clv_df):,} customers  ·  {len(filtered_txns):,} transactions")

# ---- KPI row ----

k1, k2, k3, k4, k5 = st.columns(5)

avg_clv     = clv_df["total_revenue"].mean()
median_clv  = clv_df["total_revenue"].median()
top20_share = clv_df[clv_df["clv_tier"] == "High"]["total_revenue"].sum() / clv_df["total_revenue"].sum() * 100
total_rev   = clv_df["total_revenue"].sum()
avg_orders  = clv_df["order_count"].mean()

k1.metric("Total Revenue",    f"${total_rev:,.0f}")
k2.metric("Avg CLV",          f"${avg_clv:,.0f}")
k3.metric("Median CLV",       f"${median_clv:,.0f}")
k4.metric("Top 20% Rev Share", f"{top20_share:.1f}%")
k5.metric("Avg Orders/Customer", f"{avg_orders:.1f}")

st.markdown("---")

# ---- row 1: distribution + segment bar ----

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("CLV Distribution")
    fig = px.histogram(
        clv_df, x="total_revenue", nbins=40,
        color="clv_tier",
        color_discrete_map={"High": "#2196F3", "Mid": "#66BB6A", "Low": "#EF5350"},
        labels={"total_revenue": "CLV ($)", "clv_tier": "Tier"},
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05, showlegend=True, height=340,
                      margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Avg CLV by Segment")
    seg_clv = (
        clv_df.groupby("segment")["total_revenue"]
        .mean().reset_index()
        .sort_values("total_revenue", ascending=True)
    )
    fig2 = px.bar(
        seg_clv, x="total_revenue", y="segment", orientation="h",
        labels={"total_revenue": "Avg CLV ($)", "segment": ""},
        color="total_revenue",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig2.update_layout(coloraxis_showscale=False, height=340,
                       margin=dict(t=20, b=40, l=10, r=20))
    st.plotly_chart(fig2, use_container_width=True)

# ---- row 2: monthly revenue trend + channel breakdown ----

col3, col4 = st.columns([3, 2])

with col3:
    st.subheader("Monthly Revenue Trend")
    monthly = (
        filtered_txns.groupby(["month", "segment"])["revenue"]
        .sum().reset_index()
    )
    fig3 = px.area(
        monthly, x="month", y="revenue", color="segment",
        labels={"revenue": "Revenue ($)", "month": "", "segment": "Segment"},
        template="plotly_white",
    )
    fig3.update_layout(height=340, margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("CLV by Acquisition Channel")
    ch_clv = (
        clv_df.groupby("channel")["total_revenue"]
        .agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "avg_clv", "count": "customers"})
    )
    fig4 = px.scatter(
        ch_clv, x="customers", y="avg_clv", text="channel",
        size="avg_clv", color="avg_clv",
        color_continuous_scale="Viridis",
        labels={"avg_clv": "Avg CLV ($)", "customers": "Customer Count"},
        template="plotly_white",
    )
    fig4.update_traces(textposition="top center")
    fig4.update_layout(coloraxis_showscale=False, height=340,
                       margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig4, use_container_width=True)

# ---- row 3: recency vs CLV + age group ----

col5, col6 = st.columns(2)

with col5:
    st.subheader("Recency vs CLV")
    fig5 = px.scatter(
        clv_df, x="recency_days", y="total_revenue",
        color="segment", size="order_count",
        hover_data=["customer_id", "order_count", "avg_order_value"],
        labels={"recency_days": "Days Since Last Order", "total_revenue": "CLV ($)"},
        template="plotly_white", opacity=0.7,
    )
    fig5.update_layout(height=340, margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("CLV Tier Mix by Age Group")
    age_tier = (
        clv_df.groupby(["age_group", "clv_tier"])
        .size().reset_index(name="count")
    )
    fig6 = px.bar(
        age_tier, x="age_group", y="count", color="clv_tier",
        barmode="stack",
        color_discrete_map={"High": "#2196F3", "Mid": "#66BB6A", "Low": "#EF5350"},
        labels={"count": "Customers", "age_group": "Age Group", "clv_tier": "Tier"},
        template="plotly_white",
    )
    fig6.update_layout(height=340, margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig6, use_container_width=True)

# ---- row 4: top customers table ----

st.subheader("Top 20 Customers by CLV")

top20 = (
    clv_df.nlargest(20, "total_revenue")
    [[
        "customer_id", "segment", "channel", "total_revenue",
        "order_count", "avg_order_value", "recency_days", "clv_tier"
    ]]
    .rename(columns={
        "customer_id":    "Customer",
        "segment":        "Segment",
        "channel":        "Channel",
        "total_revenue":  "CLV ($)",
        "order_count":    "Orders",
        "avg_order_value":"Avg Order ($)",
        "recency_days":   "Days Since Last Order",
        "clv_tier":       "Tier",
    })
    .reset_index(drop=True)
)
top20.index += 1

st.dataframe(top20, use_container_width=True)

st.markdown("---")
st.caption("Data is synthetically generated for demonstration. CLV = sum of historical revenue within the selected date range.")
