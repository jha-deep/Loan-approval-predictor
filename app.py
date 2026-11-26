from pathlib import Path
import sqlite3
from datetime import datetime
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "replace-this-secret-before-deployment"
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "loan_model.joblib"
DB_PATH = ROOT / "applications.db"

FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

def db_conn():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with db_conn() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prediction TEXT NOT NULL,
                approval_probability REAL NOT NULL
            )
        """)

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Run: python train_model.py")
    return joblib.load(MODEL_PATH)

@app.route("/")
def index():
    with db_conn() as db:
        total = db.execute("SELECT COUNT(*) n FROM applications").fetchone()["n"]
        approved = db.execute("SELECT COUNT(*) n FROM applications WHERE prediction='Approved'").fetchone()["n"]
        recent = db.execute("SELECT * FROM applications ORDER BY id DESC LIMIT 6").fetchall()
    rate = round(100 * approved / total, 1) if total else 0
    return render_template("index.html", total=total, approved=approved, rate=rate, recent=recent)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            raw = {
                "Gender": request.form["Gender"],
                "Married": request.form["Married"],
                "Dependents": request.form["Dependents"],
                "Education": request.form["Education"],
                "Self_Employed": request.form["Self_Employed"],
                "ApplicantIncome": float(request.form["ApplicantIncome"]),
                "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
                "LoanAmount": float(request.form["LoanAmount"]),
                "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
                "Credit_History": float(request.form["Credit_History"]),
                "Property_Area": request.form["Property_Area"]
            }
            numeric = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
            if any(raw[k] < 0 for k in numeric) or raw["Loan_Amount_Term"] == 0 or raw["Credit_History"] not in (0, 1):
                raise ValueError("Check income, loan, term and credit history values.")
            X = pd.DataFrame([raw], columns=FEATURES)
            model = load_model()
            pred = str(model.predict(X)[0])
            classes = list(model.classes_)
            probability = float(model.predict_proba(X)[0][classes.index("Y")])
            label = "Approved" if pred == "Y" else "Not Approved"
            with db_conn() as db:
                db.execute(
                    "INSERT INTO applications(created_at,prediction,approval_probability) VALUES(?,?,?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M"), label, probability)
                )
            return render_template("result.html", label=label, probability=round(probability*100, 2))
        except (ValueError, KeyError) as exc:
            flash(f"Please check the form values. {exc}", "error")
        except Exception as exc:
            flash(f"Prediction could not be completed: {exc}", "error")
    return render_template("predict.html")

@app.route("/history")
def history():
    with db_conn() as db:
        records = db.execute("SELECT * FROM applications ORDER BY id DESC").fetchall()
    return render_template("history.html", records=records)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
