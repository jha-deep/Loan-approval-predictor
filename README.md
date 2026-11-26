# Loan Approval Predictor — Educational ML Project

A Flask web application demonstrating a machine-learning classification workflow using a public historical loan prediction dataset. It includes data preprocessing, model training, a sample input form, model output page, dashboard, and local SQLite history.

> **Fairness and responsible-use notice:** This is an educational demonstration, not a credit scoring system, lender decision engine, financial advice, or a reliable assessment of any applicant. Do not use it to approve, reject, rank, price, or otherwise make decisions about real people. The sample dataset is small, historical, and not representative of current lending populations. Inputs such as gender and marital status are included only because they appear in the legacy teaching dataset; their presence is not an endorsement of using protected or sensitive characteristics in lending.

## Dataset setup

Download a public loan prediction training CSV (commonly shared as the Analytics Vidhya/Kaggle loan prediction practice dataset) and save it as:

`data/loan_prediction.csv`

The CSV is expected to include columns such as:
`Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status`


## Project structure

```text
loan-approval-predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── loan_prediction.csv   # add your licensed dataset here
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── history.html
└── static/
    └── style.css
```


## Evaluation

The training script uses a stratified holdout split and saves accuracy, confusion matrix, classification report and ROC-AUC (where available) in `metrics.json`. These are educational test-set metrics only, not evidence of suitability for lending decisions.

For any real-world lending research, substantial work would be required: lawful data governance, privacy and security review, representative data, independent validation, calibration, explainability, bias and disparate-impact assessment, monitoring, appeal processes, and qualified human oversight.
