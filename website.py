
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data"/"processed"/"catchments_clean.csv"
RESULTS = ROOT/"outputs"/"results"/"catchment_analysis_results.csv"
MODEL = ROOT/"outputs"/"models"/"flood_risk_model.pkl"

st.set_page_config(page_title="Smart Catchment Optimization", page_icon="🌧️", layout="wide")
st.title("🌧️ Smart Catchment Optimization")
st.caption("Geospatial decision-support demonstration for rainwater management")
st.info("Project demonstration using a 60-row synthetic sample dataset. The live weather panel uses Open-Meteo model data for Mylavaram town; it is not a rain gauge measurement for every catchment. Sample catchment coordinates, labels, Curve Numbers and MCDA weights are illustrative and require local validation.")

if not DATA.exists():
    st.error("Dataset missing: data/processed/catchments_clean.csv")
    st.stop()
raw = pd.read_csv(DATA)
df = pd.read_csv(RESULTS) if RESULTS.exists() else raw.copy()

pages = ["Dashboard", "Live Rainfall", "Catchment Analysis", "Rainfall Analysis",
         "SCS-CN Runoff", "MCDA Suitability", "Catchment Map",
         "ML Prediction", "About & Limitations"]
page = st.sidebar.radio("Navigation", pages)

if page == "Dashboard":
    a,b,c,d = st.columns(4)
    a.metric("Catchments", len(raw))
    b.metric("High synthetic risk", int((raw["flood_risk_class_synthetic"]=="High").sum()))
    c.metric("Mean sample rainfall", f'{raw["rainfall_intensity_mm_hr"].mean():.1f} mm/hr')
    d.metric("Mean area", f'{raw["area_ha"].mean():.2f} ha')
    st.subheader("Synthetic flood-risk class distribution")
    counts = raw["flood_risk_class_synthetic"].value_counts().rename_axis("Risk class").reset_index(name="Count")
    st.plotly_chart(px.pie(counts, names="Risk class", values="Count", hole=.35), use_container_width=True)
    st.subheader("Sample rainfall intensity by land use")
    st.plotly_chart(px.box(raw, x="land_use", y="rainfall_intensity_mm_hr", color="land_use"), use_container_width=True)

elif page == "Live Rainfall":
    st.subheader("Live weather and precipitation — Mylavaram")
    st.write("This panel requests the latest available model-based current conditions and hourly precipitation forecast from Open-Meteo.")
    st.caption("This is weather-model output for a point near Mylavaram town, not a direct rain-gauge observation or catchment-wide rainfall measurement.")
    if st.button("Refresh live weather", type="primary"):
        st.cache_data.clear()
    @st.cache_data(ttl=900, show_spinner=False)
    def get_mylavaram_weather():
        # Geocode the named place instead of hard-coding unverified coordinates.
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name":"Mylavaram","count":10,"language":"en","format":"json"},
            timeout=20
        )
        geo.raise_for_status()
        candidates = geo.json().get("results", [])
        # Prefer Mylavaram in Andhra Pradesh, India.
        match = next((x for x in candidates
                      if x.get("country_code") == "IN"
                      and x.get("admin1","").lower() == "andhra pradesh"), None)
        if match is None:
            raise ValueError("Could not uniquely locate Mylavaram in Andhra Pradesh, India.")
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": match["latitude"],
                "longitude": match["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,rain,showers,wind_speed_10m",
                "hourly": "precipitation,precipitation_probability,rain",
                "forecast_days": 2,
                "timezone": "Asia/Kolkata"
            },
            timeout=20
        )
        weather.raise_for_status()
        return match, weather.json()

    try:
        with st.spinner("Fetching weather data..."):
            place, weather = get_mylavaram_weather()
        current = weather.get("current", {})
        units = weather.get("current_units", {})
        st.caption(f"Location match: {place.get('name')}, {place.get('admin1')}, {place.get('country')} | "
                   f"Coordinates: {place['latitude']:.4f}, {place['longitude']:.4f}")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Temperature", f"{current.get('temperature_2m','—')} {units.get('temperature_2m','°C')}")
        c2.metric("Humidity", f"{current.get('relative_humidity_2m','—')} {units.get('relative_humidity_2m','%')}")
        c3.metric("Current precipitation", f"{current.get('precipitation','—')} {units.get('precipitation','mm')}")
        c4.metric("Rain", f"{current.get('rain','—')} {units.get('rain','mm')}")
        st.caption(f"Weather model timestamp: {current.get('time','not provided')} (Asia/Kolkata)")
        hourly = weather.get("hourly", {})
        if hourly.get("time"):
            hdf = pd.DataFrame({
                "Time": pd.to_datetime(hourly["time"]),
                "Precipitation (mm)": hourly.get("precipitation", []),
                "Rain (mm)": hourly.get("rain", []),
                "Precipitation probability (%)": hourly.get("precipitation_probability", [])
            })
            st.subheader("Next 48 hours — hourly precipitation forecast")
            st.plotly_chart(px.bar(hdf, x="Time", y="Precipitation (mm)",
                                   title="Forecast precipitation by hour"), use_container_width=True)
            st.dataframe(hdf, use_container_width=True)
            st.download_button("Download hourly weather CSV", hdf.to_csv(index=False),
                               "mylavaram_hourly_weather.csv", "text/csv")
        st.caption("Source: Open-Meteo Forecast API and Geocoding API. Forecast/model estimates can differ from local observations.")
    except Exception as e:
        st.error(f"Could not retrieve live weather data: {e}")
        st.info("Check your internet connection and try Refresh live weather. Other project modules can still be used.")

elif page == "Catchment Analysis":
    st.subheader("Explore catchments")
    choices = st.multiselect("Land use", sorted(raw.land_use.unique()), default=sorted(raw.land_use.unique()))
    view = raw[raw.land_use.isin(choices)]
    st.dataframe(view, use_container_width=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False), "filtered_catchments.csv", "text/csv")

elif page == "Rainfall Analysis":
    st.subheader("Sample dataset rainfall patterns")
    st.metric("Mean sample rainfall intensity", f'{raw.rainfall_intensity_mm_hr.mean():.2f} mm/hr')
    st.plotly_chart(px.histogram(raw, x="rainfall_intensity_mm_hr", nbins=12,
                                 title="Distribution of sample rainfall intensity"), use_container_width=True)
    st.plotly_chart(px.scatter(raw, x="area_ha", y="rainfall_intensity_mm_hr",
                               color="land_use", hover_name="catchment_id",
                               title="Area versus sample rainfall"), use_container_width=True)

elif page == "SCS-CN Runoff":
    st.subheader("SCS Curve Number runoff — demonstration")
    if "scs_cn_runoff_depth_mm_demo" not in df:
        st.warning("Run src/run_analysis.py first.")
    else:
        st.metric("Total demo runoff volume", f'{df.scs_cn_runoff_volume_m3_demo.sum():,.1f} m³')
        st.dataframe(df[["catchment_id","demo_hsg_assumption","demo_curve_number",
                         "rainfall_intensity_mm_hr","scs_cn_runoff_depth_mm_demo",
                         "scs_cn_runoff_volume_m3_demo"]], use_container_width=True)
        st.caption("Rainfall intensity is treated as a one-hour event depth. Soil-to-HSG mapping and CN values are provisional; validate against local soil/HSG and rainfall data.")

elif page == "MCDA Suitability":
    st.subheader("Multi-Criteria Decision Analysis")
    if "mcda_priority_score_demo" not in df:
        st.warning("Run src/run_analysis.py first.")
    else:
        st.caption("Illustrative weights: rainfall 35%, slope 20%, imperviousness 30%, drainage 15%.")
        st.plotly_chart(px.bar(df.sort_values("mcda_priority_score_demo", ascending=False),
                               x="catchment_id", y="mcda_priority_score_demo",
                               color="mcda_priority_class_demo",
                               title="MCDA priority score by catchment"), use_container_width=True)
        st.dataframe(df.sort_values("mcda_priority_rank_demo")[
            ["catchment_id","mcda_priority_score_demo","mcda_priority_rank_demo","mcda_priority_class_demo"]],
            use_container_width=True)

elif page == "Catchment Map":
    st.subheader("Sample coordinate map")
    st.warning("Coordinates are synthetic and must not be interpreted as actual Mylavaram catchment locations.")
    fig = px.scatter_mapbox(raw, lat="latitude_synthetic", lon="longitude_synthetic",
                            color="flood_risk_class_synthetic", hover_name="catchment_id",
                            hover_data=["land_use","rainfall_intensity_mm_hr"],
                            zoom=9, height=600)
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

elif page == "ML Prediction":
    st.subheader("Random Forest prediction")
    if not MODEL.exists():
        st.warning("Model not found. Run: python src/train_model.py")
    else:
        model = joblib.load(MODEL)
        with st.form("prediction"):
            land = st.selectbox("Land use", sorted(raw.land_use.unique()))
            soil = st.selectbox("Soil type", sorted(raw.soil_type.unique()))
            drainage = st.selectbox("Drainage capacity", ["Low","Medium","High"])
            area = st.number_input("Area (ha)", min_value=0.01, value=3.0)
            slope = st.number_input("Slope (%)", min_value=0.0, value=3.0)
            imperv = st.slider("Impervious surface (%)", 0.0, 100.0, 50.0)
            rain = st.number_input("Rainfall intensity (mm/hr)", min_value=0.0, value=40.0)
            storage = st.number_input("Storage capacity (m³)", min_value=0.0, value=50.0)
            runoff_c = st.slider("Assumed runoff coefficient", 0.0, 1.0, 0.5)
            submit = st.form_submit_button("Predict risk class")
        if submit:
            sample = pd.DataFrame([{
                "land_use":land, "area_ha":area, "slope_percent":slope,
                "impervious_surface_percent":imperv,
                "rainfall_intensity_mm_hr":rain, "soil_type":soil,
                "drainage_capacity_class":drainage, "storage_capacity_m3":storage,
                "runoff_coefficient_assumed":runoff_c
            }])
            pred = model.predict(sample)[0]
            st.success(f"Model output: {pred}")
            st.caption("This is a model output learned from synthetic labels, not a validated flood forecast.")

else:
    st.subheader("Project scope and limitations")
    st.markdown("""
    **Implemented**
    - Exploratory analysis of the 60-row sample dataset.
    - Random Forest classification trained on synthetic labels.
    - Provisional SCS-CN runoff calculation.
    - Weighted-sum MCDA ranking using illustrative weights.
    - Plotly map of sample coordinates.
    - Live model-based weather/precipitation panel for Mylavaram town using Open-Meteo.

    **Not yet implemented / not validated**
    - Rain-gauge observations or sensor network integration.
    - Verified Mylavaram boundary, DEM, land-cover, and soil rasters.
    - Locally verified hydrologic soil groups and Curve Numbers.
    - Calibration against observed streamflow/runoff/flood records.
    - Stakeholder-validated MCDA weights.

    Do not use this educational prototype for engineering design, emergency decisions, or public safety.
    """)
