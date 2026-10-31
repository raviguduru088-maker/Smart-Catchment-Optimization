
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "catchments_clean.csv"
MODEL_DIR = ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
target = "flood_risk_class_synthetic"
features = [
    "land_use", "area_ha", "slope_percent",
    "impervious_surface_percent", "rainfall_intensity_mm_hr",
    "soil_type", "drainage_capacity_class", "storage_capacity_m3",
    "runoff_coefficient_assumed"
]
X, y = df[features], df[target]
cat = X.select_dtypes(include="object").columns.tolist()
num = X.select_dtypes(exclude="object").columns.tolist()

prep = ColumnTransformer([
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num)
])
model = Pipeline([
    ("prep", prep),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42,
                                  class_weight="balanced"))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Rows:", len(df), "| Train:", len(X_train), "| Test:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, pred, labels=model.classes_))
joblib.dump(model, MODEL_DIR / "flood_risk_model.pkl")
print("Saved:", MODEL_DIR / "flood_risk_model.pkl")
print("CAUTION: target labels are synthetic; this score is not real-world validation.")
