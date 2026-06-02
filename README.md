# CLV Dashboard

interactive customer lifetime value dashboard built with streamlit + plotly.
no database needed — generates synthetic transaction data on startup.

---

## setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

opens at http://localhost:8501

---

## what's in it

**visualizations**
- clv distribution histogram by tier (high/mid/low)
- avg clv by customer segment (champions → lost)
- monthly revenue trend stacked by segment
- acquisition channel comparison (avg clv vs customer count)
- recency vs clv scatter — size = order count
- clv tier mix by age group
- top 20 customers table

**sidebar filters**
- date range — all charts + metrics update to the selected window
- segment multiselect
- acquisition channel multiselect
- age group multiselect

---

## how clv is calculated

historical clv = sum of all revenue from a customer within the selected date range.

```
CLV = Σ (order_value) for all orders up to snapshot date
```

**tiers** are assigned by percentile:
- **High** — top 20% by revenue
- **Mid** — 50th–80th percentile
- **Low** — bottom 50%

this is a backward-looking / realized-value model. its simple, explainable,
and doesn't require assumptions about future behavior. if you need predictive
CLV (probability of future purchases × predicted spend), look into the BG/NBD
model or Pareto/NBD — the `lifetimes` python library has good implementations.

---

## deploying to streamlit cloud

1. push repo to github
2. go to share.streamlit.io
3. point it at `app.py`
4. done — free tier works fine for this

---

## data

synthetic. 200 customers, ~18 months of transactions. segment behavior
is parameterized to produce realistic distributions (champions buy more
frequently at higher AOV, lost customers have 1-2 low-value orders).
