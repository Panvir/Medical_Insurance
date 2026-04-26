from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


LOCAL_DATA_PATH = Path(__file__).with_name("insurance.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not LOCAL_DATA_PATH.exists():
        st.error(
            "Could not find insurance.csv. Place it in the same folder as app.py."
        )
        st.stop()

    data = pd.read_csv(LOCAL_DATA_PATH).drop_duplicates()
    return data


@st.cache_resource
def train_models(data: pd.DataFrame):
    features = data.drop(columns="charges")
    target = data["charges"]

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
        ),
        "Linear Regression": LinearRegression(),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    trained = {}
    metrics = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        trained[model_name] = pipeline
        metrics.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "R2 Score": r2_score(y_test, predictions),
            }
        )

    return trained, pd.DataFrame(metrics)


def money(value: float) -> str:
    return f"${value:,.0f}"


def style_page() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.75rem;
            }
            div[data-testid="stVerticalBlock"] > div:has(.section-title) {
                border-top: 1px solid #e5e7eb;
                padding-top: 1rem;
            }
            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #111827;
                margin: 0;
            }
            .subtle {
                color: #6b7280;
                font-size: 0.95rem;
            }
            .prediction-panel {
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 1.1rem 1.2rem;
                background: #ffffff;
            }
            .prediction-value {
                font-size: clamp(2rem, 5vw, 3.5rem);
                line-height: 1;
                font-weight: 800;
                color: #0f766e;
                margin: 0.2rem 0 0.6rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


style_page()
data = load_data()
models, model_metrics = train_models(data)

st.sidebar.header("Patient Profile")

model_name = st.sidebar.selectbox(
    "Prediction model",
    options=list(models.keys()),
    help="Random Forest usually captures non-linear cost patterns better for this dataset.",
)

age = st.sidebar.slider("Age", 18, 64, 32)
sex = st.sidebar.radio("Sex", ["female", "male"], horizontal=True)
bmi = st.sidebar.slider("BMI", 15.0, 55.0, 28.0, step=0.1)
children = st.sidebar.slider("Children", 0, 5, 1)
smoker = st.sidebar.radio("Smoker", ["no", "yes"], horizontal=True)
region = st.sidebar.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"],
)

profile = pd.DataFrame(
    [
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }
    ]
)

prediction = float(models[model_name].predict(profile)[0])
prediction = max(prediction, 0)

median_charge = float(data["charges"].median())
average_charge = float(data["charges"].mean())
percentile = float((data["charges"] <= prediction).mean() * 100)

st.title("Medical Insurance Cost Predictor")
st.caption(
    "Interactive machine learning demo using age, BMI, children, smoker status, sex, and region to estimate yearly medical insurance charges."
)

top_left, top_right = st.columns([1.15, 0.85], gap="large")

with top_left:
    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
    st.markdown("Estimated yearly charge")
    st.markdown(
        f'<p class="prediction-value">{money(prediction)}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="subtle">This estimate is around the {percentile:.0f}th percentile of the dataset.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Dataset Median", money(median_charge))
    metric_b.metric("Dataset Average", money(average_charge))

    selected_metrics = model_metrics.loc[model_metrics["Model"] == model_name].iloc[0]
    metric_c, metric_d = st.columns(2)
    metric_c.metric("Model MAE", money(selected_metrics["MAE"]))
    metric_d.metric("R2 Score", f"{selected_metrics['R2 Score']:.2f}")

st.markdown('<p class="section-title">Profile Summary</p>', unsafe_allow_html=True)
summary_cols = st.columns(6)
summary_items = [
    ("Age", age),
    ("Sex", sex.title()),
    ("BMI", f"{bmi:.1f}"),
    ("Children", children),
    ("Smoker", smoker.title()),
    ("Region", region.title()),
]
for column, (label, value) in zip(summary_cols, summary_items):
    column.metric(label, value)

st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)
comparison = model_metrics.copy()
comparison["MAE"] = comparison["MAE"].round(0)
comparison["RMSE"] = comparison["RMSE"].round(0)
comparison["R2 Score"] = comparison["R2 Score"].round(3)
st.dataframe(comparison, use_container_width=True, hide_index=True)

chart_left, chart_right = st.columns(2, gap="large")

with chart_left:
    st.markdown('<p class="section-title">Charges by Smoker Status</p>', unsafe_allow_html=True)
    chart = (
        alt.Chart(data)
        .mark_boxplot(size=48)
        .encode(
            x=alt.X("smoker:N", title="Smoker"),
            y=alt.Y("charges:Q", title="Charges"),
            color=alt.Color(
                "smoker:N",
                scale=alt.Scale(domain=["yes", "no"], range=["#dc2626", "#0f766e"]),
                legend=None,
            ),
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

with chart_right:
    st.markdown('<p class="section-title">Age, BMI, and Charges</p>', unsafe_allow_html=True)
    chart = (
        alt.Chart(data)
        .mark_circle(opacity=0.72)
        .encode(
            x=alt.X("age:Q", title="Age"),
            y=alt.Y("charges:Q", title="Charges"),
            size=alt.Size("bmi:Q", title="BMI", scale=alt.Scale(range=[35, 360])),
            color=alt.Color(
                "smoker:N",
                title="Smoker",
                scale=alt.Scale(domain=["yes", "no"], range=["#dc2626", "#0f766e"]),
            ),
            tooltip=["age", "bmi", "children", "smoker", "region", "charges"],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
st.dataframe(data.head(20), use_container_width=True, hide_index=True)
