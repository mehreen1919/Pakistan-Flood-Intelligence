import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Pakistan Flood Intelligence",
    page_icon="🌊",
    layout="wide"
)

REPO_RAW = (
    "https://raw.githubusercontent.com/mehreen1919/"
    "Smart-AI-Disaster-Resource-Response-System/refs/heads/main/"
)


def data_source(env_name, file_name):
    """Where to read a dataset from, in order: the path in the environment
    variable, the file next to this app (as in the GitHub repository), or
    the copy on GitHub (as in Colab)."""
    chosen = os.environ.get(env_name)

    if chosen:
        return chosen

    if os.path.exists(file_name):
        return file_name

    return REPO_RAW + file_name


FLOOD_URL = data_source(
    "FLOOD_CSV_PATH",
    "Pakistan_flood_dataset.csv"
)

RECOVERY_URL = data_source(
    "RECOVERY_CSV_PATH",
    "pakistan_asset_recovery_2025_prototype.csv"
)

# Small reference tables built from public data (2017 census population,
# district areas, 116 years of national rainfall). Both are optional: if a
# file cannot be read, the app falls back to its earlier assumptions.
DISTRICTS_URL = data_source(
    "DISTRICTS_CSV_PATH",
    "app_districts_reference.csv"
)

RAIN_SCENARIOS_URL = data_source(
    "RAIN_SCENARIOS_CSV_PATH",
    "rainfall_scenario_multipliers.csv"
)

FLOOD_NUMERIC_CANDIDATES = [
    "rain_monsoon_mm",
    "rain_peak_month_mm",
    "temp_monsoon_c",
    "humidity_monsoon_pct",
    "rs_fsm_lgbm_pakistan"
]

FLOOD_FEATURE_LABELS = {
    "rain_monsoon_mm": "Monsoon rainfall",
    "rain_peak_month_mm": "Peak-month rainfall",
    "temp_monsoon_c": "Monsoon temperature",
    "humidity_monsoon_pct": "Humidity",
    "rs_fsm_lgbm_pakistan": "Terrain susceptibility",
    "province": "Province"
}

PRIORITY_VALUES = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100
}

SCENARIO_RAIN_FACTORS = {
    "Low": 0.85,
    "Moderate": 1.0,
    "Severe": 1.3
}

USER_REQUIRED_COLUMNS = [
    "province",
    "area_name",
    "asset_type",
    "damage_severity_1_5",
    "criticality_1_5",
    "accessibility_1_5",
    "population_served_or_affected_est",
    "flood_impact_pct_est",
    "estimated_repair_cost_pkr"
]

USER_NUMERIC_COLUMNS = USER_REQUIRED_COLUMNS[3:]

USER_RESOURCE_COLUMNS = [
    "workers_required_est",
    "equipment_units_est",
    "repair_days_est"
]

USER_COLUMN_ALIASES = {
    "damage_severity": "damage_severity_1_5",
    "damage": "damage_severity_1_5",
    "criticality": "criticality_1_5",
    "accessibility": "accessibility_1_5",
    "population": "population_served_or_affected_est",
    "population_served": "population_served_or_affected_est",
    "flood_impact_pct": "flood_impact_pct_est",
    "flood_impact": "flood_impact_pct_est",
    "repair_cost_pkr": "estimated_repair_cost_pkr",
    "estimated_repair_cost": "estimated_repair_cost_pkr",
    "repair_cost": "estimated_repair_cost_pkr",
    "cost_pkr": "estimated_repair_cost_pkr",
    "workers": "workers_required_est",
    "equipment": "equipment_units_est",
    "equipment_units": "equipment_units_est",
    "repair_days": "repair_days_est",
    "district": "area_name",
    "area": "area_name"
}

MAX_USER_ASSETS = 2000
MAX_UPLOAD_BYTES = 5_000_000
USER_INDEX_START = 1_000_000

# Plain-language headings for every table. Only what the reader sees changes;
# the calculations keep using the original column names.
COLUMN_LABELS = {
    "asset_id": "Asset ID",
    "year": "Year",
    "province": "Province",
    "area_name": "Area",
    "asset_type": "Asset Type",
    "damage_severity_1_5": "Damage Level (1–5)",
    "criticality_1_5": "Criticality (1–5)",
    "accessibility_1_5": "Accessibility (1–5)",
    "population_served_or_affected_est": "People Served/Affected (Est.)",
    "flood_impact_pct_est": "Flood Impact (%)",
    "estimated_repair_cost_pkr": "Estimated Repair Cost (PKR)",
    "workers_required_est": "Workers Needed (Est.)",
    "equipment_units_est": "Equipment Units (Est.)",
    "repair_days_est": "Repair Time (Days, Est.)",
    "recovery_priority_score_derived": "Priority Score (Dataset)",
    "recovery_priority_label_derived": "Priority Level (Dataset)",
    "record_type": "Record Type",
    "source_basis": "Data Source",
    "data_source": "Source",
    "flood_pct": "Flooded Area (%)",
    "flood": "Flood Recorded",
    "rain_monsoon_mm": "Monsoon Rainfall (mm)",
    "rain_peak_month_mm": "Peak-Month Rainfall (mm)",
    "temp_monsoon_c": "Monsoon Temperature (°C)",
    "humidity_monsoon_pct": "Monsoon Humidity (%)",
    "rs_fsm_lgbm_pakistan": "Terrain Flood Susceptibility",
    "ai_priority": "AI Priority Level",
    "ai_priority_score": "AI Priority Score",
    "start_day": "Start Day",
    "finish_day": "Finish Day"
}

_UNIT_WORDS = {"pct": "(%)", "mm": "(mm)", "c": "(°C)", "pkr": "(PKR)"}
_ACRONYMS = {"id": "ID", "ai": "AI", "pkr": "PKR", "ndma": "NDMA"}


def pretty_name(name):
    """A reader-friendly heading for a column name."""
    text = str(name)

    if text in COLUMN_LABELS:
        return COLUMN_LABELS[text]

    # Already written for people (spaces or capitals): leave it alone.
    if " " in text or any(char.isupper() for char in text):
        return text

    parts = [part for part in text.split("_") if part]
    unit = ""

    if len(parts) >= 3 and parts[-2:] == ["1", "5"]:
        parts = parts[:-2]
        unit = "(1–5)"

    if len(parts) > 1 and parts[-1] == "est":
        parts = parts[:-1]

    if not unit and len(parts) > 1 and parts[-1] in _UNIT_WORDS:
        unit = _UNIT_WORDS[parts[-1]]
        parts = parts[:-1]

    words = [_ACRONYMS.get(part, part.capitalize()) for part in parts]

    if not words:
        return text

    return " ".join(words + ([unit] if unit else []))


def friendly_columns(frame):
    """Copy of a table with reader-friendly column headings."""
    existing = set(frame.columns)
    used = set()
    mapping = {}

    for column in frame.columns:
        label = pretty_name(column)

        if label in used or (label != column and label in existing):
            label = column

        used.add(label)
        mapping[column] = label

    return frame.rename(columns=mapping)

st.markdown("""
<style>
.stApp {
    background-color: #f5f9fd;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #123a6d;
}
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #dce8f3;
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    background-color: #145da0;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #0c4a85;
    color: white;
}
.info-card {
    background-color: white;
    border: 1px solid #dce8f3;
    padding: 20px;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_flood_data():
    data = pd.read_csv(FLOOD_URL)

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    data["province"] = data["province"].astype(str).str.strip()
    data["area_name"] = data["area_name"].astype(str).str.strip()

    return data


@st.cache_data
def load_recovery_data():
    try:
        data = pd.read_csv(RECOVERY_URL)
        source = "GitHub"
    except Exception:
        try:
            data = pd.read_csv("/content/pakistan_asset_recovery_2025_prototype.csv")
            source = "Colab"
        except Exception:
            return None, None

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    data["province"] = data["province"].astype(str).str.strip()
    data["area_name"] = data["area_name"].astype(str).str.strip()

    return data, source


@st.cache_data(show_spinner=False)
def load_district_population():
    """Census 2017 population and area per district, keyed by area name.
    Returns {} if the file is missing, so the app keeps working."""
    try:
        table = pd.read_csv(DISTRICTS_URL)
        table["district"] = table["district"].astype(str).str.strip()
        table["Total_pop"] = pd.to_numeric(table["Total_pop"], errors="coerce")
        table["area_km2"] = pd.to_numeric(table["area_km2"], errors="coerce")
        table = table.dropna(subset=["Total_pop"])
        return {
            row["district"]: {
                "population": float(row["Total_pop"]),
                "area_km2": float(row["area_km2"])
            }
            for _, row in table.iterrows()
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_rain_factors():
    """Low / Moderate / Severe rainfall factors from the 1901-2016 national
    July-September record: 25th, 50th and 90th percentile, each divided by
    the median so Moderate stays at 1.0. Falls back to the earlier fixed
    factors if the file is missing."""
    default = {"Low": 0.85, "Moderate": 1.0, "Severe": 1.3}
    try:
        table = pd.read_csv(RAIN_SCENARIOS_URL).set_index("percentile")
        median = float(table.loc["p50", "monsoon_mm"])
        factors = {
            "Low": float(table.loc["p25", "monsoon_mm"]) / median,
            "Moderate": 1.0,
            "Severe": float(table.loc["p90", "monsoon_mm"]) / median
        }
        if not (0.3 < factors["Low"] < 1.0 < factors["Severe"] < 3.0):
            return default, False
        return {k: round(v, 2) for k, v in factors.items()}, True
    except Exception:
        return default, False


def get_average(data, column):
    if column not in data.columns:
        return 0

    values = pd.to_numeric(data[column], errors="coerce")
    value = values.mean()

    if pd.isna(value):
        return 0

    return float(value)


def area_value(data, column):
    """Average of a column, or NaN when it is missing (the model fills gaps)."""
    if column not in data.columns:
        return np.nan

    value = pd.to_numeric(data[column], errors="coerce").mean()

    if pd.isna(value):
        return np.nan

    return float(value)


def calculate_flood_risk(
    forecast_rainfall,
    forecast_humidity,
    historical_flood,
    historical_rainfall,
    susceptibility
):
    """Rule-based fallback used only when the flood model cannot be trained."""
    rain_score = min(forecast_rainfall / 500, 1)
    humidity_score = min(forecast_humidity / 100, 1)
    flood_score = min(historical_flood / 100, 1)
    rainfall_history_score = min(historical_rainfall / 1000, 1)
    susceptibility_score = min(max(susceptibility, 0), 1)

    risk = (
        rain_score * 0.35
        + humidity_score * 0.15
        + flood_score * 0.20
        + rainfall_history_score * 0.10
        + susceptibility_score * 0.20
    )

    return round(min(risk * 100, 100), 1)


# ---------------------------------------------------------------------------
# Shared modelling helpers
# ---------------------------------------------------------------------------

def build_preprocessing(categorical_features, numeric_features):
    transformers = []

    if categorical_features:
        transformers.append((
            "category",
            Pipeline([
                ("fill", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ))

    if numeric_features:
        transformers.append((
            "number",
            SimpleImputer(strategy="median"),
            numeric_features
        ))

    return ColumnTransformer(transformers=transformers)


def group_importance(pipeline):
    """Feature importance, with one-hot columns folded back into their source."""
    names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_

    grouped = {}

    for name, value in zip(names, values):
        base = name.split("__", 1)[1] if "__" in name else name

        if base.startswith("province_"):
            base = "province"

        if base.startswith("asset_type_"):
            base = "asset_type"

        grouped[base] = grouped.get(base, 0.0) + float(value)

    return pd.Series(grouped).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Flood risk model (Predict Flood Risk tab)
# ---------------------------------------------------------------------------

def make_flood_label(data):
    """Define a 'high-flood year' from the dataset. Returns (label, note)."""
    if "flood" in data.columns:
        flag = pd.to_numeric(data["flood"], errors="coerce")
        values = set(flag.dropna().unique().tolist())

        if values == {0, 1}:
            share = float(flag.mean())

            if 0.10 <= share <= 0.90:
                return flag, "the dataset's own flood flag (flood = 1)"

    if "flood_pct" in data.columns:
        percent = pd.to_numeric(data["flood_pct"], errors="coerce")
        threshold = percent.quantile(0.75)

        if pd.notna(threshold) and threshold > 0:
            label = percent.ge(threshold).astype(float).where(percent.notna())
            return (
                label,
                f"flood_pct in the top quarter of records (>= {threshold:.1f}%)"
            )

        label = percent.gt(0).astype(float).where(percent.notna())
        return label, "any recorded flooding (flood_pct > 0)"

    return None, "no usable flood column"


@st.cache_resource
def train_flood_model(data):
    try:
        frame = data.copy()

        numeric_features = [
            column
            for column in FLOOD_NUMERIC_CANDIDATES
            if column in frame.columns
        ]

        for column in numeric_features:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        numeric_features = [
            column
            for column in numeric_features
            if frame[column].notna().sum() >= 20
        ]

        if len(numeric_features) < 2:
            return {"error": "fewer than two usable weather columns"}

        label, label_note = make_flood_label(frame)

        if label is None:
            return {"error": label_note}

        categorical_features = (
            ["province"] if "province" in frame.columns else []
        )

        features = categorical_features + numeric_features

        keep = label.notna()
        X = frame.loc[keep, features]
        y = label[keep].astype(int)

        positives = int(y.sum())
        negatives = int(len(y) - positives)

        if len(y) < 40 or positives < 8 or negatives < 8:
            return {"error": "not enough records of both classes"}

        train_mask = None
        test_mask = None
        split_note = ""

        if "year" in frame.columns:
            years = pd.to_numeric(frame.loc[keep, "year"], errors="coerce")
            distinct_years = sorted(years.dropna().unique().tolist())

            if len(distinct_years) >= 5:
                test_count = max(1, int(round(len(distinct_years) * 0.2)))
                test_years = distinct_years[-test_count:]
                is_test = years.isin(test_years)
                is_train = (~is_test) & years.notna()

                if (
                    y[is_train].nunique() == 2
                    and y[is_test].nunique() == 2
                    and int(is_test.sum()) >= 15
                ):
                    train_mask = is_train
                    test_mask = is_test
                    split_note = (
                        f"trained on earlier years, tested on unseen years "
                        f"{int(test_years[0])}-{int(test_years[-1])}"
                    )

        if train_mask is None:
            train_index, test_index = train_test_split(
                X.index,
                test_size=0.25,
                stratify=y,
                random_state=42
            )
            train_mask = X.index.isin(train_index)
            test_mask = X.index.isin(test_index)
            split_note = "random 75/25 split (no usable year column)"

        def new_model():
            return Pipeline([
                (
                    "preprocessing",
                    build_preprocessing(
                        categorical_features,
                        numeric_features
                    )
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ])

        check = new_model().fit(X[train_mask], y[train_mask])
        test_probability = check.predict_proba(X[test_mask])[:, 1]
        test_prediction = (test_probability >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y[test_mask], test_probability)),
            "accuracy": float(accuracy_score(y[test_mask], test_prediction)),
            "base_rate": float(y[test_mask].mean()),
            "train_rows": int(np.sum(train_mask)),
            "test_rows": int(np.sum(test_mask))
        }

        model = new_model().fit(X, y)

        return {
            "model": model,
            "features": features,
            "numeric": numeric_features,
            "metrics": metrics,
            "label_note": label_note,
            "split_note": split_note,
            "rows": int(len(y)),
            "positive_share": float(y.mean()),
            "importance": group_importance(model)
        }

    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def predict_flood_risk(
    bundle,
    province,
    rainfall,
    humidity,
    peak_rain,
    temperature,
    susceptibility
):
    values = {
        "province": province,
        "rain_monsoon_mm": rainfall,
        "rain_peak_month_mm": peak_rain,
        "temp_monsoon_c": temperature,
        "humidity_monsoon_pct": humidity,
        "rs_fsm_lgbm_pakistan": susceptibility
    }

    row = pd.DataFrame([{
        column: values.get(column, np.nan)
        for column in bundle["features"]
    }])

    probability = bundle["model"].predict_proba(row)[0][1]

    return round(float(probability) * 100, 1)


def scenario_model_risk(bundle, province, history):
    """Model risk for the Low / Moderate / Severe rainfall scenarios."""
    rain = area_value(history, "rain_monsoon_mm")
    peak = area_value(history, "rain_peak_month_mm")
    humidity = area_value(history, "humidity_monsoon_pct")
    temperature = area_value(history, "temp_monsoon_c")
    susceptibility = area_value(history, "rs_fsm_lgbm_pakistan")

    if pd.isna(rain):
        return None

    result = {}

    for name, factor in SCENARIO_RAIN_FACTORS.items():
        result[name] = predict_flood_risk(
            bundle,
            province,
            rain * factor,
            humidity,
            peak * factor if pd.notna(peak) else np.nan,
            temperature,
            susceptibility
        )

    return result


# ---------------------------------------------------------------------------
# Recovery priority model (Plan Recovery tab)
# ---------------------------------------------------------------------------

def priority_scores(probabilities, classes):
    values = np.array([PRIORITY_VALUES.get(name, 0) for name in classes])
    return np.round(probabilities @ values, 1)


@st.cache_resource
def train_recovery_model(data):
    feature_columns = [
        "province",
        "asset_type",
        "damage_severity_1_5",
        "criticality_1_5",
        "accessibility_1_5",
        "population_served_or_affected_est",
        "flood_impact_pct_est",
        "estimated_repair_cost_pkr"
    ]

    target_column = "recovery_priority_label_derived"

    categorical_features = ["province", "asset_type"]

    numeric_features = [
        "damage_severity_1_5",
        "criticality_1_5",
        "accessibility_1_5",
        "population_served_or_affected_est",
        "flood_impact_pct_est",
        "estimated_repair_cost_pkr"
    ]

    labelled = data[target_column].notna()
    X = data.loc[labelled, feature_columns]
    y = data.loc[labelled, target_column]

    def new_model():
        return Pipeline([
            (
                "preprocessing",
                build_preprocessing(
                    categorical_features,
                    numeric_features
                )
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ])

    model = new_model().fit(X, y)
    classes = model.named_steps["model"].classes_

    n_splits = int(min(5, y.value_counts().min()))

    if n_splits >= 2:
        folds = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        probabilities = cross_val_predict(
            new_model(),
            X,
            y,
            cv=folds,
            method="predict_proba"
        )
        validation = (
            f"{n_splits}-fold cross-validation: every asset is scored by "
            f"a model that never saw that asset"
        )
    else:
        probabilities = model.predict_proba(X)
        validation = (
            "in-sample only (too few records per class to cross-validate)"
        )

    predicted_labels = classes[probabilities.argmax(axis=1)]
    scores = priority_scores(probabilities, classes)

    metrics = {
        "validation": validation,
        "accuracy": float(accuracy_score(y, predicted_labels)),
        "macro_f1": float(f1_score(y, predicted_labels, average="macro")),
        "rank_agreement": None,
        "importance": group_importance(model)
    }

    if "recovery_priority_score_derived" in data.columns:
        derived = pd.to_numeric(
            data.loc[labelled, "recovery_priority_score_derived"],
            errors="coerce"
        )
        predicted_scores = pd.Series(scores, index=X.index)
        agreement = predicted_scores.rank().corr(derived.rank())

        if pd.notna(agreement):
            metrics["rank_agreement"] = float(agreement)

    predictions = pd.DataFrame(
        {
            "ai_priority": predicted_labels,
            "ai_priority_score": scores
        },
        index=X.index
    )

    return model, feature_columns, predictions, metrics


def add_ai_predictions(data, model, feature_columns, predictions):
    result = data.copy()

    result["ai_priority"] = predictions["ai_priority"]
    result["ai_priority_score"] = predictions["ai_priority_score"]

    missing = result["ai_priority"].isna()

    if missing.any():
        classes = model.named_steps["model"].classes_
        probabilities = model.predict_proba(
            result.loc[missing, feature_columns]
        )
        result.loc[missing, "ai_priority"] = classes[
            probabilities.argmax(axis=1)
        ]
        result.loc[missing, "ai_priority_score"] = priority_scores(
            probabilities,
            classes
        )

    return result


# ---------------------------------------------------------------------------
# Resource allocation, scheduling and gaps
# ---------------------------------------------------------------------------

def _non_negative(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )


def optimize_resources(
    data,
    budget,
    max_workers=0,
    max_equipment=0,
    cost_column="estimated_repair_cost_pkr",
    value_column="ai_priority_score",
    workers_column="workers_required_est",
    equipment_column="equipment_units_est"
):
    """Pick the assets with the highest total priority score that fit the
    budget and, when set, the worker and equipment limits (0 = no limit).
    Uses exact costs and an exact integer solver; falls back to a greedy rule
    only if the solver is unavailable."""
    if data.empty or budget <= 0:
        return data.iloc[0:0].copy()

    items = data.copy().reset_index(drop=True)
    count = len(items)

    cost = _non_negative(items[cost_column])
    workers = _non_negative(items[workers_column])
    equipment = _non_negative(items[equipment_column])
    value = (
        pd.to_numeric(items[value_column], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
    )

    limits = [(cost, float(budget))]

    if max_workers and max_workers > 0:
        limits.append((workers, float(max_workers)))

    if max_equipment and max_equipment > 0:
        limits.append((equipment, float(max_equipment)))

    fits = np.ones(count, dtype=bool)

    for amounts, cap in limits:
        fits &= amounts <= cap

    if not fits.any():
        return items.iloc[0:0].copy()

    chosen = None

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp

        rows = []
        upper = []

        for position, (amounts, cap) in enumerate(limits):
            factor = 1e6 if position == 0 else 1.0
            rows.append(amounts / factor)
            upper.append(cap / factor)

        objective = -(value - 1e-4 * cost / 1e6)

        result = milp(
            c=objective,
            constraints=LinearConstraint(
                np.vstack(rows),
                -np.inf,
                np.array(upper)
            ),
            integrality=np.ones(count),
            bounds=Bounds(np.zeros(count), fits.astype(float)),
            options={"time_limit": 20}
        )

        if result.x is not None and result.status in (0, 1):
            chosen = [
                index for index in range(count)
                if result.x[index] > 0.5
            ]

    except Exception:
        chosen = None

    if chosen is None:
        order = np.argsort(-(value / np.maximum(cost, 1)))
        chosen = []
        used = [0.0] * len(limits)

        for index in order:
            if not fits[index]:
                continue

            if all(
                used[k] + limits[k][0][index] <= limits[k][1]
                for k in range(len(limits))
            ):
                chosen.append(int(index))

                for k in range(len(limits)):
                    used[k] += limits[k][0][index]

    # Solver tolerance safety net: never hand back a plan over any limit.
    def over_limit(selection):
        return any(
            amounts[selection].sum() > cap + 1e-6
            for amounts, cap in limits
        )

    while chosen and over_limit(chosen):
        weakest = min(
            chosen,
            key=lambda index: value[index] / max(cost[index], 1)
        )
        chosen.remove(weakest)

    return items.loc[sorted(chosen)].copy()


def schedule_repairs(selected, max_workers=0, max_equipment=0):
    """Sequence the selected repairs by priority. With no limits everything
    starts on day 0; with limits, a job waits until enough crew and
    equipment are free."""
    jobs = selected.sort_values(
        ["ai_priority_score", "repair_days_est"],
        ascending=[False, True]
    ).copy()

    placed = []
    starts = []
    finishes = []

    for _, job in jobs.iterrows():
        duration = max(int(job["repair_days_est"]), 1)
        job_workers = float(job["workers_required_est"])
        job_equipment = float(job["equipment_units_est"])

        candidates = sorted({0} | {slot[1] for slot in placed})
        start = candidates[-1]

        for candidate in candidates:
            checkpoints = {candidate} | {
                slot[0] for slot in placed
                if candidate < slot[0] < candidate + duration
            }

            feasible = True

            for moment in checkpoints:
                busy = [slot for slot in placed if slot[0] <= moment < slot[1]]
                used_workers = sum(slot[2] for slot in busy)
                used_equipment = sum(slot[3] for slot in busy)

                if (
                    max_workers > 0
                    and used_workers + job_workers > max_workers
                ) or (
                    max_equipment > 0
                    and used_equipment + job_equipment > max_equipment
                ):
                    feasible = False
                    break

            if feasible:
                start = candidate
                break

        placed.append((start, start + duration, job_workers, job_equipment))
        starts.append(start)
        finishes.append(start + duration)

    jobs["start_day"] = starts
    jobs["finish_day"] = finishes

    return jobs


def peak_concurrent(scheduled, column):
    if scheduled.empty:
        return 0

    peak = 0.0

    for moment in scheduled["start_day"].unique():
        active = scheduled[
            (scheduled["start_day"] <= moment)
            & (scheduled["finish_day"] > moment)
        ]
        peak = max(
            peak,
            float(pd.to_numeric(active[column], errors="coerce").fillna(0).sum())
        )

    return int(peak)


def coverage_gap(
    assets,
    selected,
    cost_column,
    workers_column,
    equipment_column,
    max_workers=0,
    max_equipment=0
):
    """What is still missing to also cover every Critical / High asset."""
    urgent = assets[assets["ai_priority"].isin(["Critical", "High"])]

    covered_ids = (
        set(selected["asset_id"]) if not selected.empty else set()
    )

    uncovered = urgent[~urgent["asset_id"].isin(covered_ids)]

    extra_workers = None
    extra_equipment = None

    if max_workers and max_workers > 0:
        extra_workers = int(
            pd.to_numeric(uncovered[workers_column], errors="coerce")
            .fillna(0)
            .sum()
        )

    if max_equipment and max_equipment > 0:
        extra_equipment = int(
            pd.to_numeric(uncovered[equipment_column], errors="coerce")
            .fillna(0)
            .sum()
        )

    return {
        "count": int(len(uncovered)),
        "urgent_total": int(len(urgent)),
        "extra_budget": float(
            pd.to_numeric(uncovered[cost_column], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "extra_workers": extra_workers,
        "extra_equipment": extra_equipment
    }


def show_gap(gap):
    st.subheader("🚧 Coverage Gap")

    if gap["count"] == 0:
        st.success(
            "Every Critical and High priority asset in this selection "
            "is covered by the plan."
        )
        return

    g1, g2, g3, g4 = st.columns(4)

    g1.metric(
        "Critical/High Not Covered",
        f"{gap['count']} of {gap['urgent_total']}"
    )

    g2.metric(
        "Extra Budget Needed",
        f"PKR {gap['extra_budget']:,.0f}"
    )

    g3.metric(
        "Extra Workers",
        (
            f"{gap['extra_workers']:,}"
            if gap["extra_workers"] is not None
            else "No limit set"
        )
    )

    g4.metric(
        "Extra Equipment Units",
        (
            f"{gap['extra_equipment']:,}"
            if gap["extra_equipment"] is not None
            else "No limit set"
        )
    )

    st.caption(
        "Extra needs are what it would take to also cover the Critical and "
        "High assets left out of this plan. Actual worker needs depend on "
        "how work is sequenced."
    )


def show_schedule(scheduled):
    """Repair timeline. Falls back to a table if charting is unavailable."""
    chart_data = scheduled[
        ["asset_id", "asset_type", "ai_priority", "start_day", "finish_day"]
    ].copy()

    chart_data["Asset"] = (
        chart_data["asset_id"].astype(str)
        + " · "
        + chart_data["asset_type"].astype(str)
    )

    try:
        import altair as alt

        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("start_day:Q", title="Day (from start of work)"),
            x2=alt.X2("finish_day:Q"),
            y=alt.Y("Asset:N", sort=None, title=None),
            color=alt.Color(
                "ai_priority:N",
                title="AI Priority Level",
                scale=alt.Scale(
                    domain=["Critical", "High", "Medium", "Low"],
                    range=["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]
                )
            ),
            tooltip=[
                alt.Tooltip("asset_id:N", title="Asset ID"),
                alt.Tooltip("asset_type:N", title="Asset Type"),
                alt.Tooltip("ai_priority:N", title="AI Priority Level"),
                alt.Tooltip("start_day:Q", title="Start Day"),
                alt.Tooltip("finish_day:Q", title="Finish Day")
            ]
        ).properties(
            height=int(min(max(120, 26 * len(chart_data)), 900))
        )

        try:
            st.altair_chart(chart, width="stretch")
        except TypeError:
            st.altair_chart(chart, use_container_width=True)

    except Exception:
        st.dataframe(
            friendly_columns(
                chart_data[["Asset", "ai_priority", "start_day", "finish_day"]]
            ),
            width="stretch",
            hide_index=True
        )


def download_csv_button(label, data, file_name):
    try:
        st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime="text/csv",
            on_click="ignore"
        )
    except TypeError:
        st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime="text/csv"
        )


# ---------------------------------------------------------------------------
# Explanations, robustness, equity and budget what-if
# ---------------------------------------------------------------------------

EXPLAIN_LABELS = {
    "province": "Province",
    "asset_type": "Asset type",
    "damage_severity_1_5": "Damage severity",
    "criticality_1_5": "Criticality",
    "accessibility_1_5": "Accessibility",
    "population_served_or_affected_est": "People served/affected",
    "flood_impact_pct_est": "Flood impact",
    "estimated_repair_cost_pkr": "Repair cost"
}


def explain_asset(model, feature_columns, reference, row, samples=60):
    """How much each input moves one asset's AI score, compared with a
    typical asset. Each input is swapped for values drawn from the baseline
    data and the average change in score is measured."""
    classes = model.named_steps["model"].classes_
    row_frame = pd.DataFrame([{
        column: row[column] for column in feature_columns
    }])

    base_score = float(
        priority_scores(model.predict_proba(row_frame), classes)[0]
    )

    typical = reference[feature_columns].dropna()
    typical = typical.sample(n=min(samples, len(typical)), random_state=0)

    effects = {}

    for column in feature_columns:
        swapped = pd.concat(
            [row_frame] * len(typical),
            ignore_index=True
        )
        swapped[column] = typical[column].to_numpy()

        average = priority_scores(
            model.predict_proba(swapped),
            classes
        ).mean()

        effects[column] = float(base_score - average)

    return base_score, pd.Series(effects)


def driver_text(effects, up=True, count=2, minimum=1.0):
    if up:
        picked = effects[effects >= minimum].nlargest(count)
    else:
        picked = effects[effects <= -minimum].nsmallest(count)

    if picked.empty:
        return "-"

    return ", ".join(
        f"{EXPLAIN_LABELS.get(name, name)} ({value:+.1f})"
        for name, value in picked.items()
    )


def explain_table(selected, model, feature_columns, reference, top=5):
    rows = []

    for _, row in selected.nlargest(top, "ai_priority_score").iterrows():
        _, effects = explain_asset(
            model,
            feature_columns,
            reference,
            row
        )

        rows.append({
            "Asset ID": row["asset_id"],
            "Asset Type": row["asset_type"],
            "AI Priority Score": round(float(row["ai_priority_score"]), 1),
            "Raises Priority": driver_text(effects, up=True),
            "Lowers Priority": driver_text(effects, up=False)
        })

    return pd.DataFrame(rows)


def simulate_plan_cost(costs, budget, spread_pct, runs=3000, seed=7):
    """Range of the plan's total cost if each repair estimate is uncertain.
    Costs vary around the estimate (average unchanged) with an independent
    part per asset and a shared part, since flood-related cost surprises
    tend to move together. The spread is an assumption, not a learned value."""
    values = pd.to_numeric(pd.Series(costs), errors="coerce").fillna(0)
    values = values.clip(lower=0).to_numpy(dtype=float)

    if len(values) == 0:
        return None

    sigma = max(float(spread_pct), 0.0) / 100.0
    shared = sigma / 2.0
    rng = np.random.default_rng(seed)

    individual = rng.lognormal(-(sigma ** 2) / 2, sigma, (runs, len(values)))
    common = rng.lognormal(-(shared ** 2) / 2, shared, (runs, 1))

    totals = (values * individual * common).sum(axis=1)

    return {
        "planned": float(values.sum()),
        "p_within": float((totals <= float(budget) + 1e-6).mean()),
        "p50": float(np.percentile(totals, 50)),
        "p90": float(np.percentile(totals, 90)),
        "reserve": float(max(np.percentile(totals, 90) - values.sum(), 0))
    }


def equity_summary(assets, selected):
    """Who gets funded: Critical/High coverage by area and by accessibility
    rating. Descriptive only; it does not decide what is fair."""
    funded_ids = set(selected["asset_id"]) if not selected.empty else set()

    frame = assets.copy()
    frame["funded"] = frame["asset_id"].isin(funded_ids)
    frame["urgent"] = frame["ai_priority"].isin(["Critical", "High"])
    frame["urgent_funded"] = frame["urgent"] & frame["funded"]

    by_area = frame.groupby("area_name").agg(
        assets=("asset_id", "count"),
        urgent=("urgent", "sum"),
        funded=("funded", "sum"),
        urgent_funded=("urgent_funded", "sum")
    ).reset_index()

    unserved = by_area[
        (by_area["urgent"] > 0) & (by_area["urgent_funded"] == 0)
    ]["area_name"].tolist()

    table = None

    if len(by_area) >= 2:
        by_area["share"] = np.where(
            by_area["urgent"] > 0,
            by_area["urgent_funded"] / by_area["urgent"].replace(0, np.nan) * 100,
            np.nan
        )
        table = by_area.sort_values(
            "share",
            na_position="last"
        ).rename(columns={
            "area_name": "Area",
            "assets": "Assets",
            "urgent": "Critical/High Assets",
            "funded": "Assets Funded",
            "urgent_funded": "Critical/High Funded",
            "share": "Critical/High Funded (%)"
        })
        table["Critical/High Funded (%)"] = (
            table["Critical/High Funded (%)"].round(0)
        )

    urgent = frame[frame["urgent"]]
    harder = urgent[urgent["accessibility_1_5"] <= 2]
    easier = urgent[urgent["accessibility_1_5"] >= 4]

    access = None

    if len(harder) >= 5 and len(easier) >= 5:
        access = {
            "low_n": int(len(harder)),
            "high_n": int(len(easier)),
            "low_share": float(harder["funded"].mean()),
            "high_share": float(easier["funded"].mean())
        }

    return {"table": table, "unserved": unserved, "access": access}


def budget_sensitivity(
    assets,
    budget,
    max_workers,
    max_equipment,
    base_selected,
    factors=(0.5, 0.75, 1.0, 1.5, 2.0)
):
    """Re-run the plan at several budget levels (same resource limits)."""
    total_value = float(
        pd.to_numeric(assets["ai_priority_score"], errors="coerce")
        .fillna(0)
        .sum()
    )
    urgent_total = int(
        assets["ai_priority"].isin(["Critical", "High"]).sum()
    )

    rows = []

    for factor in factors:
        level = float(budget) * factor

        if factor == 1.0:
            chosen = base_selected
        else:
            chosen = optimize_resources(
                assets,
                level,
                max_workers,
                max_equipment
            )

        if chosen.empty:
            value, urgent_funded = 0.0, 0
        else:
            value = float(chosen["ai_priority_score"].sum())
            urgent_funded = int(
                chosen["ai_priority"].isin(["Critical", "High"]).sum()
            )

        rows.append({
            "Budget Change": f"{factor - 1:+.0%}",
            "Budget (PKR Million)": round(level / 1e6, 1),
            "Assets Funded": int(len(chosen)),
            "Critical/High Funded": urgent_funded,
            "Critical/High Assets (All)": urgent_total,
            "Priority Covered (%)": (
                round(100 * value / total_value, 1)
                if total_value > 0 else 0.0
            )
        })

    return pd.DataFrame(rows)


def quality_flags(assets, baseline):
    """Plain-language warnings for unusual user-supplied values.
    Warnings only: nothing is blocked or changed."""
    cost = pd.to_numeric(
        baseline["estimated_repair_cost_pkr"],
        errors="coerce"
    )
    low = cost.groupby(baseline["asset_type"]).quantile(0.05)
    high = cost.groupby(baseline["asset_type"]).quantile(0.95)
    people_max = pd.to_numeric(
        baseline["population_served_or_affected_est"],
        errors="coerce"
    ).max()
    known_types = set(baseline["asset_type"].dropna())
    known_provinces = set(baseline["province"].dropna())

    flags = []

    for _, row in assets.iterrows():
        notes = []
        asset_type = row["asset_type"]
        amount = float(row["estimated_repair_cost_pkr"])

        if asset_type in known_types:
            if amount > 3 * high[asset_type]:
                notes.append("cost far above usual for this asset type")
            elif amount < low[asset_type] / 3:
                notes.append("cost far below usual for this asset type")
        else:
            notes.append("asset type not in baseline data")

        if row["province"] not in known_provinces:
            notes.append("province not in baseline data")

        if row["damage_severity_1_5"] >= 4 and row["flood_impact_pct_est"] < 10:
            notes.append("severe damage but very low flood impact")

        if (
            row["criticality_1_5"] >= 4
            and row["population_served_or_affected_est"] < 500
        ):
            notes.append("high criticality but very few people served")

        if (
            pd.notna(people_max)
            and row["population_served_or_affected_est"] > 2 * people_max
        ):
            notes.append("population far above the baseline range")

        flags.append("; ".join(notes))

    return pd.Series(flags, index=assets.index)


# ---------------------------------------------------------------------------
# User-supplied assets (manual entry or CSV upload)
# ---------------------------------------------------------------------------

def _clean_text(series, lookup=None):
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lstrip("=+-@")
        .str.strip()
        .str.slice(0, 60)
    )

    if lookup:
        cleaned = cleaned.map(
            lambda value: lookup.get(value.lower(), value)
        )

    return cleaned.mask(cleaned == "")


def _to_number(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = (
        series.astype(str)
        .str.replace(r"(?i)pkr|rs\.?|[,\s]", "", regex=True)
    )

    return pd.to_numeric(text, errors="coerce")


def clean_user_assets(raw, baseline, taken_ids):
    """Validate user rows and shape them like the baseline recovery data.
    Returns (assets, notes, error)."""
    notes = []

    data = raw.copy()
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    data = data.loc[:, ~data.columns.duplicated()]

    renames = {
        old: new
        for old, new in USER_COLUMN_ALIASES.items()
        if old in data.columns and new not in data.columns
    }
    data = data.rename(columns=renames)

    missing = [
        column for column in USER_REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        return None, notes, (
            "Missing required column(s): " + ", ".join(missing)
            + ". Download the CSV template to see the expected layout."
        )

    data = data.dropna(how="all")

    if data.empty:
        return None, notes, "The file has no data rows."

    if len(data) > MAX_USER_ASSETS:
        notes.append(
            f"Only the first {MAX_USER_ASSETS:,} rows were read."
        )
        data = data.head(MAX_USER_ASSETS)

    data = data.reset_index(drop=True)
    total_rows = len(data)

    for column in ("province", "area_name", "asset_type"):
        lookup = {
            str(value).lower(): str(value)
            for value in baseline[column].dropna().unique()
        }
        data[column] = _clean_text(data[column], lookup)

    for column in USER_NUMERIC_COLUMNS:
        data[column] = _to_number(data[column])

    valid = (
        data[["province", "area_name", "asset_type"]].notna().all(axis=1)
        & data[USER_NUMERIC_COLUMNS].notna().all(axis=1)
    )

    for column in (
        "damage_severity_1_5",
        "criticality_1_5",
        "accessibility_1_5"
    ):
        valid &= data[column].between(1, 5)

    valid &= data["population_served_or_affected_est"] >= 0
    valid &= data["flood_impact_pct_est"].between(0, 100)
    valid &= data["estimated_repair_cost_pkr"] > 0

    skipped = int((~valid).sum())

    if skipped:
        notes.append(
            f"{skipped} of {total_rows} row(s) were skipped (missing "
            f"values, ratings outside 1-5, flood impact outside 0-100, "
            f"negative population, or cost not above zero)."
        )

    data = data[valid].copy()

    if data.empty:
        return None, notes, "No valid rows were found."

    for column in (
        "damage_severity_1_5",
        "criticality_1_5",
        "accessibility_1_5",
        "population_served_or_affected_est",
        "estimated_repair_cost_pkr"
    ):
        data[column] = data[column].round().astype(int)

    data["flood_impact_pct_est"] = data["flood_impact_pct_est"].round(1)

    estimated = pd.Series(False, index=data.index)

    for column in USER_RESOURCE_COLUMNS:
        if column in data.columns:
            values = _to_number(data[column])
        else:
            values = pd.Series(np.nan, index=data.index)

        values = values.where(values > 0)

        baseline_values = pd.to_numeric(
            baseline[column],
            errors="coerce"
        )
        by_type = baseline_values.groupby(baseline["asset_type"]).median()
        fallback = baseline_values.median()

        fill = (
            data["asset_type"].map(by_type)
            .fillna(fallback)
            .fillna(1)
        )

        estimated |= values.isna()

        data[column] = (
            np.ceil(values.fillna(fill)).clip(lower=1).astype(int)
        )

    if estimated.any():
        notes.append(
            f"{int(estimated.sum())} asset(s) had no workers, equipment "
            f"or repair-days values, so the baseline median for that "
            f"asset type was used."
        )

    if "asset_id" in data.columns:
        raw_ids = _clean_text(data["asset_id"]).tolist()
    else:
        raw_ids = [np.nan] * len(data)

    seen = set(str(value) for value in taken_ids)
    counter = 1
    renamed = 0
    final_ids = []

    for value in raw_ids:
        if isinstance(value, str) and value:
            base_id = value
        else:
            while f"USR-{counter:04d}" in seen:
                counter += 1
            base_id = f"USR-{counter:04d}"

        candidate = base_id
        suffix = 2

        while candidate in seen:
            candidate = f"{base_id}-{suffix}"
            suffix += 1
            renamed += 1 if suffix == 3 else 0

        seen.add(candidate)
        final_ids.append(candidate)

    if renamed:
        notes.append(
            f"{renamed} asset ID(s) already existed and were renamed."
        )

    result = data.reindex(columns=list(baseline.columns))
    result["asset_id"] = final_ids
    result["record_type"] = "user_supplied_asset_record"
    result["source_basis"] = "User-supplied"
    result["data_source"] = "User-added"

    return result.reset_index(drop=True), notes, None


def read_uploaded_csv(raw_bytes):
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("file is larger than 5 MB")

    last_error = None

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def asset_template_csv():
    sample = pd.DataFrame([
        {
            "asset_id": "NEW-001",
            "province": "Punjab",
            "area_name": "Multan",
            "asset_type": "Bridge",
            "damage_severity_1_5": 4,
            "criticality_1_5": 4,
            "accessibility_1_5": 3,
            "population_served_or_affected_est": 12000,
            "flood_impact_pct_est": 65.0,
            "estimated_repair_cost_pkr": 35000000,
            "workers_required_est": 40,
            "equipment_units_est": 4,
            "repair_days_est": 45
        },
        {
            "asset_id": "",
            "province": "Sindh",
            "area_name": "Badin",
            "asset_type": "Road",
            "damage_severity_1_5": 3,
            "criticality_1_5": 3,
            "accessibility_1_5": 2,
            "population_served_or_affected_est": 8000,
            "flood_impact_pct_est": 40.0,
            "estimated_repair_cost_pkr": 6000000,
            "workers_required_est": "",
            "equipment_units_est": "",
            "repair_days_est": ""
        }
    ])

    return sample.to_csv(index=False)


def add_user_assets(new_assets):
    """Append to the session's added assets. Returns (rows added, note)."""
    current = st.session_state.get("user_assets")
    existing = 0 if current is None else len(current)
    room = MAX_USER_ASSETS - existing

    if room <= 0:
        return 0, (
            f"The limit of {MAX_USER_ASSETS:,} added assets has been reached."
        )

    note = ""

    if len(new_assets) > room:
        new_assets = new_assets.head(room)
        note = (
            f"Only {room:,} more asset(s) fit within the "
            f"{MAX_USER_ASSETS:,}-asset limit."
        )

    if current is None:
        combined = new_assets
    else:
        combined = pd.concat([current, new_assets], ignore_index=True)

    st.session_state["user_assets"] = combined.reset_index(drop=True)

    return len(new_assets), note


def build_analysis_frame(base, user_assets, include_baseline):
    base_part = base.copy()
    base_part["data_source"] = "Baseline"

    if user_assets is None or len(user_assets) == 0:
        return base_part

    user_part = user_assets.copy()
    user_part.index = range(
        USER_INDEX_START,
        USER_INDEX_START + len(user_part)
    )

    if include_baseline:
        return pd.concat([base_part, user_part])

    return user_part


def render_asset_intake(baseline):
    with st.expander("📥 Add or upload assets (optional)"):
        st.write(
            "Add your own assets, or upload a CSV with updated data, to "
            "include them in the recovery and preparedness plans. Added "
            "assets are scored by the same AI model from the damage, "
            "criticality, accessibility, population, flood impact and cost "
            "you provide."
        )

        counter = st.session_state.get("upload_counter", 0)
        known_ids = set(baseline["asset_id"].astype(str))
        current = st.session_state.get("user_assets")

        if current is not None:
            known_ids |= set(current["asset_id"].astype(str))

        st.markdown("**Upload a CSV**")

        download_csv_button(
            "Download CSV template",
            asset_template_csv(),
            "asset_upload_template.csv"
        )

        uploaded = st.file_uploader(
            "CSV file",
            type=["csv"],
            key=f"asset_upload_{counter}"
        )

        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            signature = hashlib.md5(raw_bytes).hexdigest()

            if signature != st.session_state.get("upload_signature"):
                st.session_state["upload_signature"] = signature

                try:
                    frame = read_uploaded_csv(raw_bytes)
                    cleaned, notes, error = clean_user_assets(
                        frame,
                        baseline,
                        known_ids
                    )
                except Exception as problem:
                    cleaned, notes = None, []
                    error = (
                        "This file could not be read as a CSV "
                        f"({type(problem).__name__})."
                    )

                if error:
                    st.session_state["intake_message"] = ("error", error)
                else:
                    added, limit_note = add_user_assets(cleaned)

                    if limit_note:
                        notes.append(limit_note)

                    st.session_state["intake_message"] = (
                        "warning" if (notes or added == 0) else "success",
                        " ".join(
                            [f"Added {added:,} asset(s) from the file."]
                            + notes
                        )
                    )

                st.rerun()

        st.markdown("**Add one asset**")

        known_provinces = sorted(
            baseline["province"].dropna().astype(str).unique()
        )
        known_types = sorted(
            baseline["asset_type"].dropna().astype(str).unique()
        )

        with st.form("add_asset_form", clear_on_submit=True):
            a1, a2, a3 = st.columns(3)

            with a1:
                new_province = st.selectbox(
                    "Province of asset",
                    known_provinces,
                    key="new_asset_province"
                )
                new_damage = st.number_input(
                    "Damage severity (1-5)",
                    min_value=1, max_value=5, value=3, step=1,
                    key="new_asset_damage"
                )
                new_population = st.number_input(
                    "People served/affected",
                    min_value=0, value=5000, step=500,
                    key="new_asset_population"
                )
                new_workers = st.number_input(
                    "Workers required (0 = estimate)",
                    min_value=0, value=0, step=1,
                    key="new_asset_workers"
                )

            with a2:
                new_area = st.text_input(
                    "Area / district name",
                    max_chars=60,
                    key="new_asset_area"
                )
                new_criticality = st.number_input(
                    "Criticality (1-5)",
                    min_value=1, max_value=5, value=3, step=1,
                    key="new_asset_criticality"
                )
                new_flood = st.number_input(
                    "Flood impact (%)",
                    min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                    key="new_asset_flood"
                )
                new_equipment = st.number_input(
                    "Equipment units (0 = estimate)",
                    min_value=0, value=0, step=1,
                    key="new_asset_equipment"
                )

            with a3:
                new_type = st.selectbox(
                    "Type of asset",
                    known_types,
                    key="new_asset_type"
                )
                new_access = st.number_input(
                    "Accessibility (1-5)",
                    min_value=1, max_value=5, value=3, step=1,
                    key="new_asset_access"
                )
                new_cost = st.number_input(
                    "Estimated repair cost (PKR)",
                    min_value=0, value=10000000, step=500000,
                    key="new_asset_cost"
                )
                new_days = st.number_input(
                    "Repair days (0 = estimate)",
                    min_value=0, value=0, step=1,
                    key="new_asset_days"
                )

            submitted = st.form_submit_button("Add asset")

        if submitted:
            single = pd.DataFrame([{
                "province": new_province,
                "area_name": new_area,
                "asset_type": new_type,
                "damage_severity_1_5": new_damage,
                "criticality_1_5": new_criticality,
                "accessibility_1_5": new_access,
                "population_served_or_affected_est": new_population,
                "flood_impact_pct_est": new_flood,
                "estimated_repair_cost_pkr": new_cost,
                "workers_required_est": new_workers,
                "equipment_units_est": new_equipment,
                "repair_days_est": new_days
            }])

            cleaned, notes, error = clean_user_assets(
                single,
                baseline,
                known_ids
            )

            if error or cleaned is None:
                st.session_state["intake_message"] = (
                    "error",
                    "The asset was not added: enter an area name and a "
                    "repair cost above zero."
                )
            else:
                added, limit_note = add_user_assets(cleaned)

                st.session_state["intake_message"] = (
                    "warning" if added == 0 else "success",
                    (
                        "Asset added." if added
                        else limit_note
                    )
                )

            st.rerun()

        message = st.session_state.get("intake_message")

        if message:
            {
                "success": st.success,
                "warning": st.warning,
                "error": st.error
            }[message[0]](message[1])

        user_assets = st.session_state.get("user_assets")

        if user_assets is not None and len(user_assets):
            st.divider()

            st.markdown(f"**{len(user_assets):,} added asset(s)**")

            st.checkbox(
                "Include the baseline records in the analysis",
                value=True,
                key="include_baseline"
            )

            added_view = user_assets[[
                "asset_id",
                "province",
                "area_name",
                "asset_type",
                "damage_severity_1_5",
                "criticality_1_5",
                "accessibility_1_5",
                "population_served_or_affected_est",
                "flood_impact_pct_est",
                "estimated_repair_cost_pkr",
                "workers_required_est",
                "equipment_units_est",
                "repair_days_est"
            ]].copy()

            added_view["Data Check"] = quality_flags(user_assets, baseline)

            flagged = int((added_view["Data Check"] != "").sum())

            if flagged:
                st.warning(
                    f"{flagged} added asset(s) have unusual values (see "
                    f"the Data Check column). They are still included in "
                    f"the plan; please confirm the numbers."
                )

            st.dataframe(
                friendly_columns(added_view),
                width="stretch",
                hide_index=True
            )

            if st.button("Clear added assets", key="clear_user_assets"):
                st.session_state["user_assets"] = None
                st.session_state["upload_counter"] = counter + 1
                st.session_state["upload_signature"] = None
                st.session_state["intake_message"] = None
                st.rerun()

        st.caption(
            "Added assets stay in this browser session only and are lost on "
            "refresh. Their priority comes from the model trained on the "
            "baseline data, not from the file. If an AI briefing key is set, "
            "the plan summary (totals and the five highest-priority assets) "
            "is sent to the AI service to write the briefing."
        )


# ---------------------------------------------------------------------------
# AI situation briefing
# ---------------------------------------------------------------------------

BRIEFING_PROMPT = (
    "You write short situation briefings for flood-response planners in "
    "Pakistan. Use ONLY the figures in the JSON provided. Never invent "
    "numbers, places, dates or events, and do not add outside facts. Write "
    "130-180 words in plain language with four short parts: situation, what "
    "the plan covers, gaps and risks, suggested next steps. Remind the "
    "reader once that the underlying data is prototype/synthetic and the "
    "output supports, but does not replace, official assessments."
)


def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if key:
        return key

    try:
        return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def llm_briefing(facts_json, model_name):
    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key(), timeout=45.0)

    response = client.messages.create(
        model=model_name,
        max_tokens=600,
        system=BRIEFING_PROMPT,
        messages=[{
            "role": "user",
            "content": "Facts (JSON):\n" + facts_json
        }]
    )

    text = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", "") == "text"
    ).strip()

    if not text:
        raise ValueError("empty response")

    return text


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()

    return str(value)


def render_briefing(facts, template_text):
    with st.expander("🧠 AI Situation Briefing", expanded=True):
        text = None

        if get_api_key():
            try:
                with st.spinner("Writing briefing..."):
                    text = llm_briefing(
                        json.dumps(
                            facts,
                            sort_keys=True,
                            default=_json_default
                        ),
                        os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
                    )
            except Exception as error:
                st.caption(
                    f"AI briefing unavailable ({type(error).__name__}); "
                    "showing the standard summary instead."
                )

        st.markdown(text or template_text)

        if text:
            st.caption(
                "Written by an AI model from the figures computed above. "
                "Numbers come from the planning code, not the model. "
                "Review before use."
            )
        else:
            st.caption(
                "Standard summary generated from the computed figures. "
                "Set an ANTHROPIC_API_KEY to get an AI-written briefing."
            )


def priority_mix_text(counts):
    order = ["Critical", "High", "Medium", "Low"]
    parts = [
        f"{counts[name]} {name}"
        for name in order
        if counts.get(name, 0) > 0
    ]
    return ", ".join(parts) if parts else "none"


def recovery_template(facts):
    lines = [
        f"**Scope:** {facts['scope']}, {facts['assets_in_scope']} assets "
        f"assessed"
        + (
            f" ({facts['user_added_assets']} user-added)"
            if facts.get("user_added_assets") else ""
        )
        + ".",
        f"**Plan:** {facts['assets_selected']} assets funded with "
        f"PKR {facts['allocated_pkr']:,.0f} of PKR {facts['budget_pkr']:,.0f} "
        f"(PKR {facts['remaining_pkr']:,.0f} unspent). Priority mix: "
        f"{priority_mix_text(facts['priority_mix_selected'])}.",
        f"**Resources:** about {facts['workers_total']:,} workers and "
        f"{facts['equipment_total']:,} equipment units across all repairs; "
        f"the longest single repair takes {facts['longest_repair_days']} days"
        + (
            f", and with the limits set everything finishes by day "
            f"{facts['completion_day']}."
            if facts["limits_set"] else "."
        )
    ]

    if facts["uncovered_critical_high"] > 0:
        lines.append(
            f"**Gap:** {facts['uncovered_critical_high']} Critical/High "
            f"assets are not covered; covering them would need about "
            f"PKR {facts['extra_budget_needed_pkr']:,.0f} more."
        )
    else:
        lines.append(
            "**Gap:** every Critical and High priority asset is covered."
        )

    if facts.get("chance_within_budget_pct") is not None:
        lines.append(
            f"**Cost risk:** with the assumed cost uncertainty, the chance "
            f"the plan stays within budget is about "
            f"{facts['chance_within_budget_pct']}%; holding "
            f"PKR {facts['suggested_reserve_pkr']:,.0f} in reserve would "
            f"raise that to about 90%."
        )

    if facts.get("areas_without_urgent_funding"):
        lines.append(
            "**Coverage:** no Critical/High asset is funded in "
            + ", ".join(facts["areas_without_urgent_funding"])
            + "."
        )

    lines.append(
        "*Prototype data: use to support, not replace, official "
        "assessments.*"
    )

    return "\n\n".join(lines)


def future_template(facts):
    lines = [
        f"**Scenario:** {facts['scope']}, {facts['scenario']} flood "
        f"scenario for {facts['planning_year']}: about "
        f"{facts['scenario_flooded_pct']:.1f}% of the area flooded "
        f"(historical average {facts['historical_flooded_pct']:.1f}%)."
        + (
            f" Model-estimated chance of a high-flood year: "
            f"{facts['model_risk_pct']:.1f}%."
            if facts.get("model_risk_pct") is not None else ""
        ),
        f"**Needs:** illustrative funding need of "
        f"PKR {facts['funding_need_pkr']:,.0f}, "
        f"{facts['workers_total']:,} workers and "
        f"{facts['equipment_total']:,} equipment units (asset-level sums).",
        f"**Plan:** PKR {facts['allocated_pkr']:,.0f} of "
        f"PKR {facts['budget_pkr']:,.0f} allocated, covering "
        f"{facts['assets_covered']} of {facts['assets_total']} assets.",
        f"**Relief:** roughly {facts['people_affected']:,.0f} people "
        f"affected, needing {facts['water_litres_day']:,.0f} litres of water "
        f"and {facts['rations_day']:,.0f} food rations per day and "
        f"{facts['tents']:,.0f} family tents (editable assumptions)."
    ]

    if facts["uncovered_critical_high"] > 0:
        lines.append(
            f"**Gap:** {facts['uncovered_critical_high']} Critical/High "
            f"assets are not covered; about "
            f"PKR {facts['extra_budget_needed_pkr']:,.0f} more would be "
            f"needed."
        )
    else:
        lines.append(
            "**Gap:** every Critical and High priority asset is covered."
        )

    lines.append(
        "*Planning estimates on prototype data, not forecasts or "
        "procurement advice.*"
    )

    return "\n\n".join(lines)


flood_df = load_flood_data()
recovery_df, recovery_source = load_recovery_data()

flood_model = train_flood_model(flood_df)
flood_model_ready = isinstance(flood_model, dict) and "model" in flood_model

SCENARIO_RAIN_FACTORS, RAIN_FACTORS_FROM_HISTORY = load_rain_factors()
DISTRICT_POPULATION = load_district_population()

if recovery_df is not None:
    (
        recovery_model,
        recovery_features,
        recovery_predictions,
        recovery_metrics
    ) = train_recovery_model(recovery_df)

    analysis_df = build_analysis_frame(
        recovery_df,
        st.session_state.get("user_assets"),
        st.session_state.get("include_baseline", True)
    )

    predicted_df = add_ai_predictions(
        analysis_df,
        recovery_model,
        recovery_features,
        recovery_predictions
    )
else:
    predicted_df = None
    recovery_metrics = None

st.title("🇵🇰 Pakistan Flood Intelligence")

st.caption(
    "AI-Assisted Flood Preparedness & Recovery Planning"
)

st.write(
    "Analyze historical flood conditions, estimate flood risk, "
    "predict recovery priority and create a recovery plan "
    "within an available budget."
)

st.divider()

prediction_tab, recovery_tab, future_tab, dataset_tab = st.tabs([
    "🌊 Predict Flood Risk",
    "🔧 Plan Recovery",
    "🗓️ Future Resource Planning",
    "📊 Dataset & Model"
])


with prediction_tab:
    st.header("Flood Risk Prediction")

    st.write(
        "Select an area to view historical flood conditions "
        "and enter expected weather conditions."
    )
    st.caption(
        "Prototype data: the historical flood records have not been "
        "checked against official flood reports, and some years known to "
        "be severe (such as 2010 and 2022) do not stand out in them, so "
        "treat the estimates as a demonstration of the method."
    )

    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.subheader("📍 Select Location")

        provinces = sorted(
            flood_df["province"]
            .dropna()
            .unique()
        )

        province = st.selectbox(
            "Province",
            provinces,
            key="flood_province"
        )

        province_data = flood_df[
            flood_df["province"] == province
        ]

        areas = sorted(
            province_data["area_name"]
            .dropna()
            .unique()
        )

        area = st.selectbox(
            "Area",
            areas,
            key="flood_area"
        )

        area_data = flood_df[
            (flood_df["province"] == province)
            & (flood_df["area_name"] == area)
        ].copy()

        st.success(
            f"{len(area_data)} historical record(s) "
            f"loaded for {area}, {province}"
        )

        avg_flood_pct = get_average(
            area_data,
            "flood_pct"
        )

        avg_monsoon_rain = get_average(
            area_data,
            "rain_monsoon_mm"
        )

        avg_peak_rain = get_average(
            area_data,
            "rain_peak_month_mm"
        )

        avg_temp = get_average(
            area_data,
            "temp_monsoon_c"
        )

        avg_humidity = get_average(
            area_data,
            "humidity_monsoon_pct"
        )

        avg_susceptibility = get_average(
            area_data,
            "rs_fsm_lgbm_pakistan"
        )

        st.subheader("🌦️ Forecast Conditions")

        forecast_rainfall = st.number_input(
            "Expected Monsoon Rainfall (mm)",
            min_value=0.0,
            max_value=2000.0,
            value=float(
                round(
                    avg_monsoon_rain
                    if avg_monsoon_rain > 0
                    else 300,
                    1
                )
            ),
            step=10.0
        )

        forecast_humidity = st.number_input(
            "Expected Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(
                round(
                    avg_humidity
                    if avg_humidity > 0
                    else 70,
                    1
                )
            ),
            step=1.0
        )

        analyze = st.button(
            "Analyze Flood Risk",
            type="primary",
            width="stretch"
        )

    with right:
        st.subheader(
            f"📊 Historical Profile: {area}"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Average Flooded Area",
            f"{avg_flood_pct:.1f}%"
        )

        m2.metric(
            "Monsoon Rainfall",
            f"{avg_monsoon_rain:.1f} mm"
        )

        m3.metric(
            "Peak Month Rainfall",
            f"{avg_peak_rain:.1f} mm"
        )

        m4, m5 = st.columns(2)

        m4.metric(
            "Monsoon Temperature",
            f"{avg_temp:.1f} °C"
        )

        m5.metric(
            "Monsoon Humidity",
            f"{avg_humidity:.1f}%"
        )

        st.markdown(
            f"""
            <div class="info-card">
            <b>Province:</b> {province}<br>
            <b>Area:</b> {area}<br>
            <b>Historical records:</b> {len(area_data)}<br>
            <b>Average historical flood extent:</b>
            {avg_flood_pct:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        if analyze:
            if flood_model_ready:
                peak_base = area_value(area_data, "rain_peak_month_mm")

                if pd.notna(peak_base) and avg_monsoon_rain > 0:
                    peak_for_forecast = (
                        peak_base * forecast_rainfall / avg_monsoon_rain
                    )
                else:
                    peak_for_forecast = peak_base

                flood_risk = predict_flood_risk(
                    flood_model,
                    province,
                    forecast_rainfall,
                    forecast_humidity,
                    peak_for_forecast,
                    area_value(area_data, "temp_monsoon_c"),
                    area_value(area_data, "rs_fsm_lgbm_pakistan")
                )
            else:
                flood_risk = calculate_flood_risk(
                    forecast_rainfall,
                    forecast_humidity,
                    avg_flood_pct,
                    avg_monsoon_rain,
                    avg_susceptibility
                )

            if flood_risk >= 70:
                risk_level = "HIGH"
            elif flood_risk >= 40:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"

            st.subheader("🔮 Prototype Flood Risk")

            r1, r2 = st.columns(2)

            r1.metric(
                "Estimated Flood Risk",
                f"{flood_risk}%"
            )

            r2.metric(
                "Risk Level",
                risk_level
            )

            if risk_level == "HIGH":
                st.error(
                    "🔴 HIGH RISK — Conditions indicate "
                    "a high level of potential flood risk."
                )

            elif risk_level == "MODERATE":
                st.warning(
                    "🟠 MODERATE RISK — Flood conditions "
                    "should be monitored closely."
                )

            else:
                st.success(
                    "🟢 LOW RISK — Current inputs indicate "
                    "relatively lower flood risk."
                )

            if flood_model_ready:
                st.caption(
                    f"Chance of a high-flood year, estimated by a Random "
                    f"Forest trained on {flood_model['rows']:,} historical "
                    f"records. On held-out data it scored ROC-AUC "
                    f"{flood_model['metrics']['roc_auc']:.2f} "
                    f"({flood_model['split_note']})."
                )

                if flood_model["metrics"]["roc_auc"] < 0.6:
                    st.warning(
                        "Held-out accuracy is low for this dataset, so "
                        "treat this score as indicative only."
                    )
            else:
                st.caption(
                    "Rule-based estimate: the flood model could not be "
                    "trained from this dataset."
                )

            comparison = pd.DataFrame({
                "Condition": [
                    "Historical Average",
                    "Forecast"
                ],
                "Rainfall (mm)": [
                    round(avg_monsoon_rain, 1),
                    round(forecast_rainfall, 1)
                ]
            })

            st.subheader(
                "Forecast vs Historical Rainfall"
            )

            st.bar_chart(
                comparison.set_index("Condition")
            )

            if flood_model_ready:
                with st.expander("🔎 What drives this score?"):
                    drivers = (
                        flood_model["importance"]
                        .rename(index=FLOOD_FEATURE_LABELS)
                        .to_frame("Importance")
                    )

                    st.bar_chart(drivers)

                    st.caption(
                        "Share of the model's decisions explained by each "
                        "input, across all areas. Peak-month rainfall is "
                        "scaled with your forecast rainfall; temperature "
                        "and terrain susceptibility use this area's "
                        "historical averages."
                    )

    st.divider()

    st.subheader(
        f"Historical Records — {area}"
    )

    columns_to_show = [
        "year",
        "area_name",
        "province",
        "flood_pct",
        "rain_monsoon_mm",
        "rain_peak_month_mm",
        "temp_monsoon_c",
        "humidity_monsoon_pct",
        "flood"
    ]

    columns_to_show = [
        column
        for column in columns_to_show
        if column in area_data.columns
    ]

    historical_display = area_data[
        columns_to_show
    ].copy()

    if "year" in historical_display.columns:
        historical_display = (
            historical_display
            .sort_values("year")
        )

    st.dataframe(
        friendly_columns(historical_display),
        width="stretch",
        hide_index=True
    )

    if (
        "year" in area_data.columns
        and "flood_pct" in area_data.columns
    ):
        trend_data = area_data[
            ["year", "flood_pct"]
        ].copy()

        trend_data["year"] = pd.to_numeric(
            trend_data["year"],
            errors="coerce"
        )

        trend_data["flood_pct"] = pd.to_numeric(
            trend_data["flood_pct"],
            errors="coerce"
        )

        trend_data = (
            trend_data
            .dropna()
            .sort_values("year")
        )

        if not trend_data.empty:
            st.subheader(
                "Historical Flood Trend"
            )

            st.line_chart(
                trend_data.rename(columns={
                    "year": "Year",
                    "flood_pct": "Flooded Area (%)"
                }).set_index("Year")
            )

    st.caption(
        "Historical values come from the Pakistan flood "
        "dataset. The flood-risk percentage is a prototype "
        "decision-support score, not an official forecast."
    )


with recovery_tab:
    st.header("AI-Assisted Recovery Planning")

    st.write(
        "Use the 2025 prototype recovery dataset to predict "
        "asset priority and create a budget-constrained "
        "recovery plan."
    )

    if recovery_df is None:
        st.error(
            "Recovery dataset could not be loaded. "
            "Upload pakistan_asset_recovery_2025_prototype.csv "
            "to your GitHub repository or /content folder."
        )

    else:
        st.info(
            "The recovery records are synthetic asset-level "
            "examples calibrated to NDMA 2025 infrastructure "
            "damage information. Cost and resource fields are "
            "prototype estimates."
        )

        render_asset_intake(recovery_df)

        show_source = bool(
            (predicted_df["data_source"] != "Baseline").any()
        )

        left, right = st.columns(
            [1, 1.7],
            gap="large"
        )

        with left:
            st.subheader("📍 Recovery Area")

            recovery_provinces = sorted(
                predicted_df["province"]
                .dropna()
                .unique()
            )

            recovery_province = st.selectbox(
                "Province",
                recovery_provinces,
                key="recovery_province"
            )

            recovery_province_data = predicted_df[
                predicted_df["province"]
                == recovery_province
            ]

            recovery_areas = sorted(
                recovery_province_data["area_name"]
                .dropna()
                .unique()
                .tolist()
            )

            recovery_area = st.selectbox(
                "Area",
                ["All areas"] + recovery_areas,
                key="recovery_area"
            )

            recovery_area_data = recovery_province_data.copy()
            if recovery_area != "All areas":
                recovery_area_data = recovery_area_data[
                    recovery_area_data["area_name"] == recovery_area
                ].copy()

            st.success(
                f"{len(recovery_area_data)} recovery asset "
                f"record(s) found for {recovery_area}."
            )

            st.subheader(
                "💰 Available Recovery Budget"
            )

            budget = st.number_input(
                "Budget (PKR)",
                min_value=0,
                value=100000000,
                step=1000000
            )

            with st.expander("⚙️ Resource limits (optional)"):
                max_workers = st.number_input(
                    "Available workers (0 = no limit)",
                    min_value=0,
                    value=0,
                    step=10,
                    key="recovery_workers"
                )

                max_equipment = st.number_input(
                    "Available equipment units (0 = no limit)",
                    min_value=0,
                    value=0,
                    step=1,
                    key="recovery_equipment"
                )

                cost_spread = st.number_input(
                    "Repair-cost uncertainty (± %, for the robustness check)",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=5.0,
                    key="recovery_cost_spread"
                )

            generate = st.button(
                "Generate AI Recovery Plan",
                type="primary",
                width="stretch"
            )

        with right:
            st.subheader(
                f"🏗️ Assets in {recovery_area}"
            )

            area_columns = [
                "asset_id",
                "asset_type",
                "damage_severity_1_5",
                "criticality_1_5",
                "population_served_or_affected_est",
                "estimated_repair_cost_pkr",
                "ai_priority",
                "ai_priority_score"
            ]

            if show_source:
                area_columns.append("data_source")

            area_display = recovery_area_data[area_columns].copy()

            area_display = area_display.rename(
                columns={
                    "asset_id": "Asset ID",
                    "asset_type": "Asset Type",
                    "damage_severity_1_5": "Damage Level (1–5)",
                    "criticality_1_5": "Criticality (1–5)",
                    "population_served_or_affected_est":
                        "People Served/Affected",
                    "estimated_repair_cost_pkr":
                        "Estimated Repair Cost",
                    "ai_priority": "AI Priority Level",
                    "ai_priority_score": "AI Priority Score",
                    "data_source": "Source"
                }
            )

            area_display[
                "Estimated Repair Cost"
            ] = area_display[
                "Estimated Repair Cost"
            ].map(
                lambda value:
                f"PKR {value:,.0f}"
            )

            area_display = area_display.sort_values(
                "AI Priority Score",
                ascending=False
            )

            st.dataframe(
                area_display,
                width="stretch",
                hide_index=True
            )

        if generate:
            selected_assets = optimize_resources(
                recovery_area_data,
                budget,
                max_workers,
                max_equipment
            )

            total_cost = (
                selected_assets[
                    "estimated_repair_cost_pkr"
                ].sum()
                if not selected_assets.empty
                else 0
            )

            remaining_budget = budget - total_cost

            total_workers = (
                selected_assets[
                    "workers_required_est"
                ].sum()
                if not selected_assets.empty
                else 0
            )

            total_equipment = (
                selected_assets[
                    "equipment_units_est"
                ].sum()
                if not selected_assets.empty
                else 0
            )

            limits_set = bool(max_workers > 0 or max_equipment > 0)
            completion_day = 0

            st.divider()
            st.subheader(
                "🎯 Recommended Recovery Plan"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Available Budget",
                f"PKR {budget:,.0f}"
            )

            c2.metric(
                "Allocated",
                f"PKR {total_cost:,.0f}"
            )

            c3.metric(
                "Remaining",
                f"PKR {remaining_budget:,.0f}"
            )

            c4.metric(
                "Assets Selected",
                len(selected_assets)
            )

            if selected_assets.empty:
                st.warning(
                    "The available budget is too small "
                    "for the assets in this area."
                )

            else:
                scheduled = schedule_repairs(
                    selected_assets,
                    max_workers,
                    max_equipment
                )

                completion_day = int(scheduled["finish_day"].max())

                resource1, resource2, resource3 = (
                    st.columns(3)
                )

                resource1.metric(
                    "Estimated Workers",
                    f"{int(total_workers):,}"
                )

                resource2.metric(
                    "Equipment Units",
                    f"{int(total_equipment):,}"
                )

                resource3.metric(
                    "Longest Repair Time",
                    (
                        f"{int(selected_assets['repair_days_est'].max())} "
                        "days"
                    )
                )

                if limits_set:
                    st.caption(
                        f"With the limits you set, repairs are sequenced by "
                        f"priority and all finish by day {completion_day}. "
                        f"Peak use at once: "
                        f"{peak_concurrent(scheduled, 'workers_required_est'):,} "
                        f"workers, "
                        f"{peak_concurrent(scheduled, 'equipment_units_est'):,} "
                        f"equipment units."
                    )

                plan_columns = [
                    "asset_id",
                    "asset_type",
                    "ai_priority",
                    "ai_priority_score",
                    "estimated_repair_cost_pkr",
                    "workers_required_est",
                    "equipment_units_est",
                    "repair_days_est",
                    "start_day",
                    "finish_day"
                ]

                if show_source:
                    plan_columns.append("data_source")

                plan_display = scheduled[plan_columns].copy()

                plan_display = plan_display.rename(
                    columns={
                        "asset_id": "Asset ID",
                        "asset_type": "Asset Type",
                        "ai_priority": "AI Priority Level",
                        "ai_priority_score": "AI Priority Score",
                        "estimated_repair_cost_pkr":
                            "Estimated Cost",
                        "workers_required_est": "Workers Needed",
                        "equipment_units_est":
                            "Equipment Units",
                        "repair_days_est": "Repair Time (Days)",
                        "start_day": "Start Day",
                        "finish_day": "Finish Day",
                        "data_source": "Source"
                    }
                )

                plan_display[
                    "Estimated Cost"
                ] = plan_display[
                    "Estimated Cost"
                ].map(
                    lambda value:
                    f"PKR {value:,.0f}"
                )

                plan_display = plan_display.sort_values(
                    "AI Priority Score",
                    ascending=False
                )

                st.dataframe(
                    plan_display,
                    width="stretch",
                    hide_index=True
                )

                st.success(
                    "The plan selects the combination of "
                    "assets that gives the highest total "
                    "AI priority score within the budget"
                    + (
                        " and resource limits."
                        if limits_set else "."
                    )
                )

                with st.expander("📅 Repair schedule"):
                    show_schedule(scheduled)

                    st.caption(
                        "Set the worker or equipment limits above to see "
                        "repairs sequenced when crews are limited."
                        if not limits_set
                        else "Bars show when each repair starts and "
                        "finishes under your worker and equipment limits."
                    )

            gap = coverage_gap(
                recovery_area_data,
                selected_assets,
                "estimated_repair_cost_pkr",
                "workers_required_est",
                "equipment_units_est",
                max_workers,
                max_equipment
            )

            show_gap(gap)

            robust = None
            equity = None

            if not selected_assets.empty:
                robust = simulate_plan_cost(
                    selected_assets["estimated_repair_cost_pkr"],
                    budget,
                    cost_spread
                )
                equity = equity_summary(
                    recovery_area_data,
                    selected_assets
                )

                with st.expander("🔎 Why these assets? (top 5)"):
                    st.dataframe(
                        explain_table(
                            selected_assets,
                            recovery_model,
                            recovery_features,
                            recovery_df
                        ),
                        width="stretch",
                        hide_index=True
                    )

                    st.caption(
                        "Each number is how many AI-score points that input "
                        "adds (+) or removes (-) for this asset compared "
                        "with a typical asset in the baseline data. It "
                        "explains the model's reasoning, which comes from "
                        "the prototype labels, not a verified judgement."
                    )

                with st.expander("🎲 Plan robustness (cost uncertainty)"):
                    q1, q2, q3 = st.columns(3)

                    q1.metric(
                        "Chance Cost Stays Within Budget",
                        f"{robust['p_within']:.0%}"
                    )

                    q2.metric(
                        "Likely Cost (Median)",
                        f"PKR {robust['p50']:,.0f}"
                    )

                    q3.metric(
                        "Cost To Be 90% Sure",
                        f"PKR {robust['p90']:,.0f}"
                    )

                    if robust["p_within"] < 0.9:
                        st.info(
                            f"The plan uses most of the budget, so cost "
                            f"overruns on a few repairs could exceed it. "
                            f"Holding about PKR {robust['reserve']:,.0f} "
                            f"in reserve would make the plan stay within "
                            f"budget in about 9 of 10 simulated cases."
                        )
                    else:
                        st.success(
                            "The plan stays within budget in at least 9 of "
                            "10 simulated cases."
                        )

                    st.caption(
                        f"3,000 simulations with each repair cost varying "
                        f"by about ±{cost_spread:.0f}% around its estimate "
                        f"(part shared across assets, as flood-related "
                        f"surprises tend to move together). The uncertainty "
                        f"level is your assumption, set under Resource "
                        f"limits; it is not learned from data."
                    )

                with st.expander("📈 Budget what-if"):
                    with st.spinner("Comparing budget levels..."):
                        what_if = budget_sensitivity(
                            recovery_area_data,
                            budget,
                            max_workers,
                            max_equipment,
                            selected_assets
                        )

                    st.dataframe(
                        what_if,
                        width="stretch",
                        hide_index=True
                    )

                    st.line_chart(
                        what_if.set_index("Budget (PKR Million)")[
                            ["Priority Covered (%)"]
                        ]
                    )

                    st.caption(
                        "The same plan re-optimised at lower and higher "
                        "budgets, with your worker and equipment limits "
                        "unchanged. Priority covered is the share of the "
                        "total AI priority score in this area that the "
                        "funded assets cover."
                    )

                with st.expander("⚖️ Equity check"):
                    if equity["table"] is not None:
                        st.dataframe(
                            equity["table"],
                            width="stretch",
                            hide_index=True
                        )

                    if equity["unserved"]:
                        st.warning(
                            "No Critical/High asset is funded in: "
                            + ", ".join(equity["unserved"][:10])
                            + (
                                f" and {len(equity['unserved']) - 10} more"
                                if len(equity["unserved"]) > 10 else ""
                            )
                            + "."
                        )
                    else:
                        st.success(
                            "Every area with Critical/High assets has at "
                            "least one funded."
                        )

                    if equity["access"] is not None:
                        gap_points = abs(
                            equity["access"]["low_share"]
                            - equity["access"]["high_share"]
                        ) * 100

                        st.caption(
                            f"Critical/High assets rated accessibility 1-2 "
                            f"({equity['access']['low_n']}): "
                            f"{equity['access']['low_share']:.0%} funded. "
                            f"Rated 4-5 ({equity['access']['high_n']}): "
                            f"{equity['access']['high_share']:.0%} funded."
                            + (
                                " The difference is large; check that this "
                                "is intended."
                                if gap_points >= 15 else ""
                            )
                        )

                    st.caption(
                        "This shows who the plan reaches. It does not decide "
                        "what is fair; the planner does."
                    )

            top_assets = []

            if not selected_assets.empty:
                for _, row in selected_assets.nlargest(
                    5,
                    "ai_priority_score"
                ).iterrows():
                    top_assets.append({
                        "asset_type": row["asset_type"],
                        "area": row["area_name"],
                        "priority": row["ai_priority"],
                        "cost_pkr": float(row["estimated_repair_cost_pkr"]),
                        "people": int(
                            row["population_served_or_affected_est"]
                        )
                    })

            recovery_facts = {
                "scope": f"{recovery_area}, {recovery_province}",
                "assets_in_scope": int(len(recovery_area_data)),
                "user_added_assets": int(
                    (recovery_area_data["data_source"] != "Baseline").sum()
                ),
                "budget_pkr": float(budget),
                "allocated_pkr": float(total_cost),
                "remaining_pkr": float(remaining_budget),
                "assets_selected": int(len(selected_assets)),
                "priority_mix_selected": (
                    selected_assets["ai_priority"].value_counts().to_dict()
                    if not selected_assets.empty else {}
                ),
                "workers_total": int(total_workers),
                "equipment_total": int(total_equipment),
                "longest_repair_days": (
                    int(selected_assets["repair_days_est"].max())
                    if not selected_assets.empty else 0
                ),
                "limits_set": limits_set,
                "worker_limit": int(max_workers) if max_workers else None,
                "equipment_limit": (
                    int(max_equipment) if max_equipment else None
                ),
                "completion_day": completion_day,
                "uncovered_critical_high": gap["count"],
                "extra_budget_needed_pkr": gap["extra_budget"],
                "chance_within_budget_pct": (
                    round(robust["p_within"] * 100) if robust else None
                ),
                "suggested_reserve_pkr": (
                    robust["reserve"] if robust else None
                ),
                "areas_without_urgent_funding": (
                    equity["unserved"][:10] if equity else []
                ),
                "top_assets": top_assets,
                "data_note": "prototype synthetic data"
            }

            render_briefing(
                recovery_facts,
                recovery_template(recovery_facts)
            )

        st.caption(
            "AI priority is predicted by a Random Forest "
            "trained on derived prototype recovery labels and scored "
            "with cross-validation. Resource allocation uses exact "
            "optimization, not AI. "
            "This is an MVP decision-support prototype."
        )


with future_tab:
    st.header("🗓️ Future Flood Resource Planning (2027–2030)")
    st.write(
        "Explore possible flood preparedness needs under adjustable scenarios. "
        "Results are planning estimates, not validated future predictions."
    )

    if recovery_df is None:
        st.warning(
            "The synthetic recovery dataset is needed for this demo. "
            "Upload pakistan_asset_recovery_2025_prototype.csv to GitHub "
            "or to the Colab /content folder."
        )
    else:
        f1, f2, f3 = st.columns(3)
        future_provinces = sorted(flood_df["province"].dropna().unique())
        with f1:
            future_province = st.selectbox(
                "Province", future_provinces, key="future_province"
            )
        province_history = flood_df[
            flood_df["province"] == future_province
        ]
        # Only offer areas represented in both datasets.
        province_assets = predicted_df[
            predicted_df["province"] == future_province
        ]
        historical_areas = set(province_history["area_name"].dropna())
        asset_areas = set(province_assets["area_name"].dropna())
        future_areas = sorted(historical_areas & asset_areas)
        with f2:
            future_area = st.selectbox(
                "Area", ["All areas"] + future_areas, key="future_area"
            )
        if not future_areas:
            st.caption(
                "No area-level overlap between flood history and synthetic "
                "recovery records for this province. All areas uses the "
                "province-level records when available."
            )
        with f3:
            future_year = st.selectbox(
                "Planning year", [2026, 2027, 2028, 2029, 2030],
                key="future_year"
            )

        f4, f5 = st.columns(2)
        with f4:
            future_scenario = st.selectbox(
                "Flood scenario", ["Low", "Moderate", "Severe"],
                index=1, key="future_scenario"
            )
        with f5:
            future_budget = st.number_input(
                "Available preparedness budget (PKR)",
                min_value=0, value=100000000, step=5000000,
                key="future_budget"
            )

        st.caption(
            "Flooded area is the historical average scaled by the scenario: "
            "Low ×0.75, Moderate ×1.00, Severe ×1.50. Repair costs and "
            "resource needs scale with it. Repair costs only change with the "
            "planning year if you set an annual escalation below."
        )

        with st.expander("⚙️ Resource limits & planning assumptions (optional)"):
            e1, e2 = st.columns(2)
            with e1:
                future_workers = st.number_input(
                    "Available workers (0 = no limit)",
                    min_value=0, value=0, step=10, key="future_workers"
                )
                future_escalation = st.number_input(
                    "Annual repair-cost escalation (%)",
                    min_value=0.0, max_value=50.0, value=0.0, step=0.5,
                    key="future_escalation"
                )
                future_shelter_pct = st.number_input(
                    "Affected people needing shelter (%)",
                    min_value=0.0, max_value=100.0, value=15.0, step=1.0,
                    key="future_shelter_pct"
                )
                future_supply_days = st.number_input(
                    "Days of relief supply to pre-position",
                    min_value=1, value=14, step=1, key="future_supply_days"
                )
            with e2:
                future_equipment = st.number_input(
                    "Available equipment units (0 = no limit)",
                    min_value=0, value=0, step=1, key="future_equipment"
                )
                future_water_lpd = st.number_input(
                    "Water per person per day (litres)",
                    min_value=1.0, value=15.0, step=1.0,
                    key="future_water_lpd"
                )
                future_tent_size = st.number_input(
                    "People per family tent",
                    min_value=1, value=5, step=1, key="future_tent_size"
                )

        if st.button("Generate preparedness plan", key="future_generate"):
            history = province_history.copy()
            if future_area != "All areas":
                history = history[history["area_name"] == future_area]

            history_flood = pd.to_numeric(
                history["flood_pct"], errors="coerce"
            ).dropna()
            if history_flood.empty:
                st.error("No usable historical flood percentages for this area.")
            else:
                baseline = float(history_flood.mean())
                multiplier = {"Low": 0.75, "Moderate": 1.0, "Severe": 1.5}[
                    future_scenario
                ]
                scenario_flood_pct = min(100.0, max(0.0, baseline * multiplier))

                assets = predicted_df[
                    predicted_df["province"] == future_province
                ].copy()
                if future_area != "All areas":
                    assets = assets[assets["area_name"] == future_area]

                st.subheader(f"Preparedness scenario: {future_area}, {future_year}")
                a, b, c = st.columns(3)
                a.metric("Historical average flooded area", f"{baseline:.1f}%")
                b.metric("Scenario flooded area", f"{scenario_flood_pct:.1f}%")
                c.metric("Available budget", f"PKR {future_budget:,.0f}")

                model_risk = None

                if flood_model_ready:
                    model_risk = scenario_model_risk(
                        flood_model, future_province, history
                    )

                if model_risk:
                    st.caption(
                        f"Model-estimated chance of a high-flood year: "
                        f"Low {model_risk['Low']:.1f}% · "
                        f"Moderate {model_risk['Moderate']:.1f}% · "
                        f"Severe {model_risk['Severe']:.1f}% "
                        f"(assuming monsoon rainfall ×"
                        f"{SCENARIO_RAIN_FACTORS['Low']:.2f} / ×1.00 / ×"
                        f"{SCENARIO_RAIN_FACTORS['Severe']:.2f} of the "
                        f"historical average"
                        + (", set from the 1901–2016 national Jul–Sep rainfall "
                           "record: 25th percentile, median, 90th percentile"
                           if RAIN_FACTORS_FROM_HISTORY else "")
                        + f"). Selected scenario: "
                        f"{model_risk[future_scenario]:.1f}%."
                    )

                if assets.empty:
                    st.info(
                        "No synthetic asset records match this selection. "
                        "Try All areas or another province."
                    )
                else:
                    # Scale synthetic needs relative to the area's historical baseline.
                    # This is a transparent scenario rule, not a learned prediction.
                    scale = scenario_flood_pct / max(baseline, 1.0)
                    escalation_factor = (
                        (1 + future_escalation / 100)
                        ** max(future_year - 2025, 0)
                    )
                    assets["scenario_cost_pkr"] = (
                        pd.to_numeric(
                            assets["estimated_repair_cost_pkr"],
                            errors="coerce"
                        ).fillna(0).clip(lower=0) * scale * escalation_factor
                    ).round(0)
                    assets["scenario_workers"] = np.ceil(
                        pd.to_numeric(
                            assets["workers_required_est"],
                            errors="coerce"
                        ).fillna(0).clip(lower=0) * scale
                    ).astype(int)
                    assets["scenario_equipment"] = np.ceil(
                        pd.to_numeric(
                            assets["equipment_units_est"],
                            errors="coerce"
                        ).fillna(0).clip(lower=0) * scale
                    ).astype(int)
                    assets["priority"] = pd.to_numeric(
                        assets["ai_priority_score"],
                        errors="coerce"
                    ).fillna(0)

                    total_cost = float(assets["scenario_cost_pkr"].sum())
                    total_workers = int(assets["scenario_workers"].sum())
                    total_equipment = int(assets["scenario_equipment"].sum())

                    st.subheader("Estimated resources for the full scenario")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Illustrative funding need", f"PKR {total_cost:,.0f}")
                    r2.metric("Workers (asset-level sum)", f"{total_workers:,}")
                    r3.metric("Equipment units (asset-level sum)", f"{total_equipment:,}")

                    # Same exact optimizer as the recovery plan: highest total
                    # priority within the budget and any worker/equipment limits.
                    selected = optimize_resources(
                        assets,
                        future_budget,
                        future_workers,
                        future_equipment,
                        cost_column="scenario_cost_pkr",
                        value_column="priority",
                        workers_column="scenario_workers",
                        equipment_column="scenario_equipment"
                    )
                    allocated = (
                        float(selected["scenario_cost_pkr"].sum())
                        if not selected.empty else 0.0
                    )
                    remaining = float(future_budget) - allocated

                    st.subheader("Budget-constrained allocation")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Budget allocated", f"PKR {allocated:,.0f}")
                    p2.metric("Budget remaining", f"PKR {remaining:,.0f}")
                    p3.metric("Assets covered", f"{len(selected)} / {len(assets)}")

                    if selected.empty:
                        st.info("No assets fit within the available budget.")
                    else:
                        display_columns = [
                            "area_name", "asset_type", "ai_priority",
                            "scenario_cost_pkr", "scenario_workers",
                            "scenario_equipment", "priority"
                        ]
                        plan_table = selected[display_columns].rename(columns={
                            "area_name": "Area",
                            "asset_type": "Asset Type",
                            "ai_priority": "AI Priority Level",
                            "scenario_cost_pkr": "Estimated Cost (PKR)",
                            "scenario_workers": "Workers Needed",
                            "scenario_equipment": "Equipment Units",
                            "priority": "AI Priority Score"
                        })
                        st.dataframe(
                            plan_table,
                            width="stretch",
                            hide_index=True
                        )
                        download_csv_button(
                            "Download preparedness plan (CSV)",
                            plan_table.to_csv(index=False),
                            (
                                f"preparedness_{future_province}_{future_year}_"
                                f"{future_scenario.lower()}.csv"
                            )
                        )

                    future_gap = coverage_gap(
                        assets,
                        selected,
                        "scenario_cost_pkr",
                        "scenario_workers",
                        "scenario_equipment",
                        future_workers,
                        future_equipment
                    )

                    show_gap(future_gap)

                    exposed_people = float(
                        pd.to_numeric(
                            assets["population_served_or_affected_est"],
                            errors="coerce"
                        ).fillna(0).sum()
                    )
                    population_basis = "asset"
                    scope_areas = sorted(assets["area_name"].unique())
                    census_known = [
                        a for a in scope_areas if a in DISTRICT_POPULATION
                    ]
                    # Use census population only when every area in scope has
                    # a figure; otherwise keep the asset-based estimate so
                    # areas are never mixed on different bases.
                    if scope_areas and len(census_known) == len(scope_areas):
                        exposed_people = float(sum(
                            DISTRICT_POPULATION[a]["population"]
                            for a in scope_areas
                        ))
                        population_basis = "census"
                    people_affected = exposed_people * scenario_flood_pct / 100
                    people_sheltered = people_affected * future_shelter_pct / 100
                    family_tents = float(
                        np.ceil(people_sheltered / max(future_tent_size, 1))
                    )
                    water_litres_day = people_affected * future_water_lpd
                    rations_day = people_affected

                    with st.expander("🧰 Relief supplies to pre-position (indicative)"):
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("People affected (est.)", f"{people_affected:,.0f}")
                        s2.metric("Water per day (litres)", f"{water_litres_day:,.0f}")
                        s3.metric("Food rations per day", f"{rations_day:,.0f}")
                        s4.metric("Family tents", f"{family_tents:,.0f}")

                        st.caption(
                            f"For {int(future_supply_days)} days: "
                            f"{water_litres_day * future_supply_days:,.0f} litres "
                            f"of water and {rations_day * future_supply_days:,.0f} "
                            f"food rations. People affected = exposed population "
                            f"× scenario flooded share. "
                            + (
                                f"Exposed population is the 2017 census total "
                                f"for the selected area(s) "
                                f"({exposed_people:,.0f}), and the flooded "
                                f"share is applied evenly across it, so "
                                f"treat the result as indicative. "
                                if population_basis == "census" else
                                f"Exposed population adds up asset-level "
                                f"estimates, which can overlap, so treat it "
                                f"as an upper bound (no census figure is "
                                f"available for every selected area). "
                            )
                            + f"The water figure uses the "
                            f"commonly cited humanitarian minimum of 15 litres "
                            f"per person per day; the other rates are "
                            f"assumptions you can edit above."
                        )

                    st.info(
                        "Method: historical average flood percentage × scenario "
                        "multiplier. Synthetic 2025 asset costs, workers and "
                        "equipment are scaled by the same factor (costs also by "
                        "any annual escalation you set). Asset selection uses "
                        "the same exact optimizer as the recovery plan, ranked "
                        "by cross-validated AI priority scores. These values "
                        "are illustrative, not forecasts, procurement "
                        "recommendations, or verified resource inventories. "
                        "Workers/equipment sums do not account for sharing "
                        "resources across assets."
                    )

                    future_facts = {
                        "scope": (
                            f"{future_area}, {future_province}"
                        ),
                        "scenario": future_scenario,
                        "planning_year": int(future_year),
                        "historical_flooded_pct": baseline,
                        "scenario_flooded_pct": scenario_flood_pct,
                        "model_risk_pct": (
                            model_risk[future_scenario] if model_risk else None
                        ),
                        "funding_need_pkr": total_cost,
                        "workers_total": total_workers,
                        "equipment_total": total_equipment,
                        "budget_pkr": float(future_budget),
                        "allocated_pkr": allocated,
                        "assets_covered": int(len(selected)),
                        "assets_total": int(len(assets)),
                        "people_affected": people_affected,
                        "water_litres_day": water_litres_day,
                        "rations_day": rations_day,
                        "tents": family_tents,
                        "uncovered_critical_high": future_gap["count"],
                        "extra_budget_needed_pkr": future_gap["extra_budget"],
                        "data_note": "prototype synthetic data"
                    }

                    render_briefing(
                        future_facts,
                        future_template(future_facts)
                    )


with dataset_tab:
    st.header("Dataset & Model Information")

    st.subheader("🌊 Flood Dataset")

    f1, f2, f3 = st.columns(3)

    f1.metric(
        "Flood Records",
        f"{len(flood_df):,}"
    )

    f2.metric(
        "Provinces",
        flood_df["province"].nunique()
    )

    f3.metric(
        "Areas",
        flood_df["area_name"].nunique()
    )

    st.dataframe(
        friendly_columns(flood_df.head(50)),
        width="stretch",
        hide_index=True
    )

    st.subheader("🤖 Flood Risk Model")

    if flood_model_ready:
        flood_metrics = flood_model["metrics"]

        fm1, fm2, fm3 = st.columns(3)

        fm1.metric(
            "Training Records",
            f"{flood_model['rows']:,}"
        )

        fm2.metric(
            "Held-out ROC-AUC",
            f"{flood_metrics['roc_auc']:.2f}",
            help=(
                "How well the model ranks flood years above non-flood "
                "years on records it never saw during training. 0.50 is "
                "no better than guessing; 1.00 is perfect."
            )
        )

        fm3.metric(
            "Held-out Accuracy",
            f"{flood_metrics['accuracy']:.0%}",
            help=(
                "Share of unseen records where the model's flood / "
                "no-flood call was correct."
            )
        )

        st.write(
            "Model: Random Forest Classifier"
        )

        st.write(
            f"Target: high-flood year, defined as "
            f"{flood_model['label_note']}"
        )

        st.write(
            "Inputs: "
            + ", ".join(
                FLOOD_FEATURE_LABELS.get(name, name).lower()
                for name in flood_model["features"]
            )
            + "."
        )

        st.caption(
            f"Validation: {flood_model['split_note']} "
            f"({flood_metrics['train_rows']:,} training and "
            f"{flood_metrics['test_rows']:,} test records; "
            f"{flood_metrics['base_rate']:.0%} of test records are "
            f"high-flood years). ROC-AUC of 0.5 would be no better than "
            f"guessing."
        )
    else:
        reason = (
            flood_model.get("error", "unknown reason")
            if isinstance(flood_model, dict) else "unknown reason"
        )

        st.warning(
            "The flood model could not be trained, so the Predict Flood "
            f"Risk tab uses its rule-based estimate. Reason: {reason}."
        )

    st.divider()
    st.subheader("🔧 Recovery Dataset")

    if recovery_df is not None:
        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Recovery Records",
            f"{len(recovery_df):,}"
        )

        r2.metric(
            "Provinces",
            recovery_df["province"].nunique()
        )

        r3.metric(
            "Asset Types",
            recovery_df["asset_type"].nunique()
        )

        st.write(
            f"Recovery dataset loaded from: "
            f"**{recovery_source}**"
        )

        st.dataframe(
            friendly_columns(recovery_df.head(50)),
            width="stretch",
            hide_index=True
        )

        st.subheader("🤖 Recovery AI Model")

        st.write(
            "Model: Random Forest Classifier"
        )

        st.write(
            "Target: recovery_priority_label_derived"
        )

        st.write(
            "Inputs: province, asset type, damage "
            "severity, criticality, accessibility, population, "
            "flood impact and estimated repair cost."
        )

        if recovery_metrics:
            q1, q2, q3 = st.columns(3)

            q1.metric(
                "Validated Accuracy",
                f"{recovery_metrics['accuracy']:.0%}",
                help=(
                    "Share of assets given the right priority level by a "
                    "model that had not seen that asset."
                )
            )

            q2.metric(
                "Macro F1",
                f"{recovery_metrics['macro_f1']:.2f}",
                help=(
                    "An accuracy score that treats every priority level "
                    "(Low to Critical) equally, so rare levels count as "
                    "much as common ones. 1.00 is perfect."
                )
            )

            q3.metric(
                "Rank Agreement",
                (
                    f"{recovery_metrics['rank_agreement']:.2f}"
                    if recovery_metrics["rank_agreement"] is not None
                    else "n/a"
                ),
                help=(
                    "How closely the AI's ordering of assets matches the "
                    "dataset's own priority ordering. 1.00 means identical "
                    "order."
                )
            )

            st.caption(
                f"Validation: {recovery_metrics['validation']}. "
                f"Rank agreement compares the AI score's ordering with the "
                f"dataset's derived priority score."
            )

            st.info(
                "The priority labels in this dataset are derived from the "
                "same inputs the model sees, so the model mainly learns "
                "that scoring rule. Its value here is a consistent ranking "
                "with class probabilities; with real field-assessed "
                "priorities, the same pipeline would learn from actual "
                "decisions."
            )

        st.warning(
            "The recovery target, repair costs and resource "
            "requirements are prototype-derived estimates. "
            "They are not verified individual government "
            "recovery decisions."
        )

st.divider()

st.caption(
    "Pakistan Flood Intelligence MVP • "
    "Decision-support prototype • "
    "Not an official flood warning, engineering assessment "
    "or government recovery decision"
)
