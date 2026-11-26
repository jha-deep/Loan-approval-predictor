from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_prediction.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Put the Kaggle Loan Prediction dataset at data/loan_prediction.csv. "
        "See README.md for the expected file."
    )

df = pd.read_csv(DATA_PATH)
target = "Loan_Status"
if target not in df.columns:
    raise ValueError("Dataset must contain the Loan_Status target column.")

# Loan_ID is an identifier, not a predictive feature.
X = df.drop(columns=[target, "Loan_ID"], errors="ignore")
y = df[target].astype(str).str.strip()

# Normalize common target encodings.
y = y.replace({"1": "Y", "0": "N", "Yes": "Y", "No": "N"})
valid = y.isin(["Y", "N"])
X, y = X.loc[valid].copy(), y.loc[valid].copy()

categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric),
    ("categorical", categorical_pipe, categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced",
        min_samples_leaf=2
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, list(model.classes_).index("Y")]

metrics = {
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "confusion_matrix": confusion_matrix(y_test, pred, labels=["N", "Y"]).tolist(),
    "classification_report": classification_report(
        y_test, pred, labels=["N", "Y"],
        target_names=["Not Approved", "Approved"], output_dict=True, zero_division=0
    )
}
if len(set(y_test)) == 2:
    metrics["roc_auc"] = round(float(roc_auc_score((y_test == "Y").astype(int), proba)), 4)

joblib.dump({"model": model, "features": list(X.columns)}, ROOT / "loan_model.joblib")
(ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved loan_model.joblib")
print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
