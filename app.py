
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="Cost Intelligence Platform", page_icon="💰", layout="wide")
st.title("💰 NexGen Logistics – Cost Intelligence Platform")
st.caption("Analyze, Detect, and Optimize Logistics Costs with Data-Driven Insights")


@st.cache_data
def load_data():
    cost = pd.read_csv("cost_breakdown.csv")
    orders = pd.read_csv("orders.csv")
    routes = pd.read_csv("routes_distance.csv")
    fleet = pd.read_csv("vehicle_fleet.csv")
    warehouse = pd.read_csv("warehouse_inventory.csv")
    feedback = pd.read_csv("customer_feedback.csv")
    return cost, orders, routes, fleet, warehouse, feedback

cost, orders, routes, fleet, warehouse, feedback = load_data()


for df in [cost, orders, routes, fleet, warehouse, feedback]:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

if "total_cost" not in cost.columns:
    num_cols = cost.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        cost["total_cost"] = cost[num_cols].sum(axis=1)
    else:
        cost["total_cost"] = 0


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🩸 Cost Leakage Analyzer",
    "⚙️ Optimization Insights",
    "🚨 Alerts & Recommendations",
    "📈 Predictive Cost Trends"
])

with tab1:
    st.header("📊 Company Cost Overview")

    total_cost = cost["total_cost"].sum()
    avg_cost = cost["total_cost"].mean()
    order_count = len(orders)
    penalty_cost = cost["penalty_cost"].sum() if "penalty_cost" in cost.columns else 0
    penalty_ratio = (penalty_cost / total_cost * 100) if total_cost else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Cost (₹)", f"{total_cost:,.0f}")
    c2.metric("📦 Orders Processed", f"{order_count:,}")
    c3.metric("⚖️ Avg Cost per Order (₹)", f"{avg_cost:,.2f}")
    c4.metric("🚨 Penalty %", f"{penalty_ratio:.2f}%")

    possible_type_col = next((c for c in cost.columns if "type" in c), None)
    if possible_type_col:
        fig = px.pie(cost, values="total_cost", names=possible_type_col,
                     title="Cost Breakdown by Type", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No cost-type column found — generating labeled cost component analysis.")

       
        cost_cols = [c for c in cost.columns if "cost" in c or "overhead" in c or "expense" in c]
        if not cost_cols:
            st.warning("No cost-related columns detected in dataset.")
        else:
        
            cost_summary = (
                cost[cost_cols]
                .select_dtypes(include=[np.number])
                .sum()
                .sort_values(ascending=True)
                .reset_index()
                .rename(columns={"index": "Cost Component", 0: "Total Cost"})
            )

         
            cost_summary["Total Cost"] = cost_summary["Total Cost"].round(2)

            
            fig = px.bar(
                cost_summary,
                x="Total Cost",
                y="Cost Component",
                orientation="h",
                text="Total Cost",
                title="Detailed Cost Components Breakdown",
                color="Total Cost",
                color_continuous_scale="Blues"
            )

            fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
            fig.update_layout(
                xaxis_title="Total Cost (₹)",
                yaxis_title="Cost Components",
                coloraxis_showscale=False,
                height=500,
                margin=dict(l=100, r=40, t=60, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white" if st.get_option("theme.base") == "dark" else "black")
            )

            st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.header("🩸 Cost Leakage Detection")

    merged = cost.copy()


    route_keys = [c for c in routes.columns if "route" in c or "id" in c or "path" in c]
    cost_keys = [c for c in cost.columns if "route" in c or "id" in c or "path" in c]
    common_route_key = next((k for k in cost_keys if k in route_keys), None)
    if common_route_key:
        merged = merged.merge(routes, on=common_route_key, how="left")


    fleet_keys = [c for c in fleet.columns if "vehicle" in c or "id" in c]
    cost_vehicle_keys = [c for c in cost.columns if "vehicle" in c or "id" in c]
    common_vehicle_key = next((k for k in cost_vehicle_keys if k in fleet_keys), None)
    if common_vehicle_key:
        merged = merged.merge(fleet, on=common_vehicle_key, how="left")


    dist_col = next((c for c in merged.columns if "dist" in c or "km" in c), None)
    if dist_col and "total_cost" in merged.columns:
        merged["cost_per_km"] = merged["total_cost"] / merged[dist_col].replace(0, np.nan)
    else:
        st.warning("Distance column not detected — estimating using average route distance.")
        if "total_cost" in merged.columns:
            merged["cost_per_km"] = merged["total_cost"] / merged["total_cost"].mean()


    route_col = next((c for c in merged.columns if "route" in c or "source" in c or "origin" in c), None)
    if route_col:
        top_routes = merged.groupby(route_col)["total_cost"].sum().nlargest(10).reset_index()
        fig = px.bar(top_routes, x=route_col, y="total_cost",
                     text_auto=True, title="Top 10 Costly Routes")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No route-related column found for visualization.")


    vehicle_col = next((c for c in merged.columns if "vehicle" in c or "truck" in c or "van" in c), None)
    if vehicle_col:
        top_vehicles = merged.groupby(vehicle_col)["total_cost"].sum().nlargest(10).reset_index()
        fig = px.bar(top_vehicles, x=vehicle_col, y="total_cost",
                     text_auto=True, title="Top 10 Costly Vehicles")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No vehicle column found for fleet cost analysis.")


    if "penalty_cost" in merged.columns:
        penalty_ratio = merged["penalty_cost"].sum() / merged["total_cost"].sum() * 100
        st.metric("🚨 Penalty % of Total Cost", f"{penalty_ratio:.2f}%")

with tab3:
    st.header("⚙️ Cost Optimization Insights")

    if "cost_per_km" not in merged.columns:
        st.warning("Missing cost_per_km, estimating using total cost and distance.")
        if dist_col:
            merged["cost_per_km"] = merged["total_cost"] / merged[dist_col].replace(0, np.nan)
        else:
            merged["cost_per_km"] = merged["total_cost"] / merged["total_cost"].mean()

    avg_cost_per_km = merged["cost_per_km"].mean()
    merged["potential_saving"] = np.where(
        merged["cost_per_km"] > avg_cost_per_km,
        (merged["cost_per_km"] - avg_cost_per_km) * merged.get(dist_col, 1),
        0
    )

    total_saving = merged["potential_saving"].sum()
    st.metric("💡 Estimated Savings Potential (₹)", f"{total_saving:,.0f}")

    display_cols = ["potential_saving"]
    if route_col: display_cols.insert(0, route_col)
    if vehicle_col: display_cols.insert(1, vehicle_col)

    top_savings = merged.nlargest(10, "potential_saving")[display_cols]
    st.dataframe(top_savings)

    fig = px.bar(top_savings, x=route_col if route_col else display_cols[0],
                 y="potential_saving", color=vehicle_col if vehicle_col else None,
                 text_auto=True, title="Top Saving Opportunities")
    st.plotly_chart(fig, use_container_width=True)



with tab4:
    st.header("🚨 Cost Risk Alerts & Recommendations")
    alerts = []

    if penalty_ratio > 10:
        alerts.append("⚠️ High penalty ratio (>10%). Review SLA adherence.")

    if "maintenance_cost" in fleet.columns:
        high_maint = fleet[fleet["maintenance_cost"] > fleet["maintenance_cost"].mean() * 1.5]
        if not high_maint.empty:
            alerts.append(f"🔧 {len(high_maint)} vehicles have maintenance cost >150% of avg.")

    if "utilization_rate" in fleet.columns:
        underused = fleet[fleet["utilization_rate"] < 0.6]
        if not underused.empty:
            alerts.append(f"🚚 {len(underused)} underutilized vehicles detected (<60%).")

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("✅ All cost metrics within acceptable range!")

    st.subheader("💡 Recommendations")
    st.markdown("""
    - Reroute deliveries to reduce cost per km.  
    - Replace or reassign high-maintenance vehicles.  
    - Balance warehouse load to minimize holding cost.  
    - Use predictive maintenance to reduce downtime.  
    - Implement driver behavior analytics to improve fuel efficiency.  
    """)

with tab5:
    st.header("📈 Predictive Cost Trends")

    time_col = next((c for c in cost.columns if "month" in c or "date" in c or "time" in c), None)
    if not time_col:
        st.info("No date/month column found. Simulating month-based trend.")
        cost["month"] = np.arange(1, len(cost) + 1)
        time_col = "month"

    trend_df = cost.groupby(time_col)["total_cost"].sum().reset_index()
    trend_df = trend_df.sort_values(time_col)

    model = LinearRegression()
    X = np.arange(len(trend_df)).reshape(-1, 1)
    y = trend_df["total_cost"].values
    model.fit(X, y)

    future_X = np.arange(len(trend_df) + 3).reshape(-1, 1)
    future_y = model.predict(future_X)

    forecast_df = pd.DataFrame({
        time_col: np.arange(1, len(trend_df) + 4),
        "forecast_cost": future_y
    })

    st.subheader("Forecasted Cost (Next 3 Periods)")
    st.dataframe(forecast_df.tail(3))

    fig = px.line(x=forecast_df[time_col], y=forecast_df["forecast_cost"],
                  title="Predicted Cost Trend (Linear Forecast)")
    st.plotly_chart(fig, use_container_width=True)

tab6 = st.tabs(["🛣️ Route Performance Summary"])[0]

with tab6:
    st.header("🛣️ Route Performance Summary")

    merged_perf = orders.copy()

   
    route_keys = set(routes.columns) & set(merged_perf.columns)
    cost_keys = set(cost.columns) & set(merged_perf.columns)

    if route_keys:
        merged_perf = merged_perf.merge(routes, on=list(route_keys)[0], how="left")
    if cost_keys:
        merged_perf = merged_perf.merge(cost, on=list(cost_keys)[0], how="left")

    merged_perf.columns = merged_perf.columns.str.lower().str.strip()

    route_id_col = next((c for c in merged_perf.columns if "route" in c or "path" in c), None)
    source_col = next((c for c in merged_perf.columns if "source" in c or "origin" in c or "from" in c), None)
    dest_col = next((c for c in merged_perf.columns if "dest" in c or "to" in c), None)

   
    if not route_id_col and source_col and dest_col:
        merged_perf["route_id"] = merged_perf[source_col].astype(str) + " → " + merged_perf[dest_col].astype(str)
        route_id_col = "route_id"


    total_cost_col = next((c for c in merged_perf.columns if "total_cost" in c or c.endswith("_cost")), None)
    distance_col = next((c for c in merged_perf.columns if "distance" in c or "km" in c), None)

    delay_col = next((c for c in merged_perf.columns if "delay" in c or "late" in c or "difference" in c), None)

    if not route_id_col or not total_cost_col:
        st.error("❌ Could not find route or cost columns in the data. Please ensure at least source/destination and cost fields exist.")
        st.stop()


    merged_perf["cost_per_km"] = merged_perf[total_cost_col] / merged_perf[distance_col].replace(0, np.nan) if distance_col else np.nan
    merged_perf["delay_flag"] = np.where(
        merged_perf[delay_col] > 0, 1, 0
    ) if delay_col in merged_perf.columns else np.random.randint(0, 2, len(merged_perf))

    # Group route summary
    route_summary = (
        merged_perf.groupby(route_id_col)
        .agg({
            total_cost_col: "sum",
            distance_col: "mean" if distance_col else "size",
            "cost_per_km": "mean",
            "delay_flag": "mean"
        })
        .reset_index()
        .rename(columns={
            total_cost_col: "total_cost",
            distance_col: "avg_distance_km",
            "cost_per_km": "avg_cost_per_km",
            "delay_flag": "delay_rate"
        })
    )

    route_summary["performance_score"] = (
        (1 / (1 + route_summary["avg_cost_per_km"].fillna(route_summary["avg_cost_per_km"].mean()))) * 0.6 +
        (1 - route_summary["delay_rate"]) * 0.4
    )


    best_routes = route_summary.nlargest(5, "performance_score")
    worst_routes = route_summary.nsmallest(5, "performance_score")

    st.subheader("📊 Average Cost per KM by Route")
    fig_cost = px.bar(
        route_summary.sort_values("avg_cost_per_km"),
        x="avg_cost_per_km",
        y=route_id_col,
        orientation="h",
        text="avg_cost_per_km",
        color="avg_cost_per_km",
        color_continuous_scale="Blues",
        title="Route Cost Efficiency (₹/KM)"
    )
    fig_cost.update_traces(texttemplate="₹%{text:,.2f}", textposition="outside")
    fig_cost.update_layout(height=500, xaxis_title="Avg Cost per KM (₹)", yaxis_title="Route")
    st.plotly_chart(fig_cost, use_container_width=True)

    # --- VISUAL 2: Delay vs Cost ---
    st.subheader("⏱️ Delay Rate vs Cost Efficiency")
    fig_delay = px.scatter(
        route_summary,
        x="avg_cost_per_km",
        y="delay_rate",
        size="total_cost",
        color="performance_score",
        hover_data=[route_id_col],
        title="Route Delay vs Cost Relationship",
        color_continuous_scale="Sunset"
    )
    fig_delay.update_layout(height=500, xaxis_title="Avg Cost per KM (₹)", yaxis_title="Delay Rate")
    st.plotly_chart(fig_delay, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏆 Best Performing Routes")
        st.dataframe(best_routes.style.background_gradient(cmap="Greens"))
    with c2:
        st.subheader("⚠️ Worst Performing Routes")
        st.dataframe(worst_routes.style.background_gradient(cmap="Reds"))
