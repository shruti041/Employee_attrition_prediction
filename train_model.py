"""Train the attrition model used by final_app.py.

Logistic Regression (one-hot encoded categories + scaled numbers) on the 16
questions that matter most. Features were chosen by forward selection: starting
from nothing, repeatedly add the question that most improves cross-validated
log-loss. Beyond ~16 questions, each extra one adds almost no accuracy.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["OverTime", "JobRole", "MaritalStatus", "BusinessTravel"]
NUMERIC = [
    "EnvironmentSatisfaction", "JobSatisfaction", "JobInvolvement",
    "WorkLifeBalance", "RelationshipSatisfaction",
    "Age", "DistanceFromHome", "NumCompaniesWorked", "TrainingTimesLastYear",
    "YearsInCurrentRole", "YearsWithCurrManager", "YearsSinceLastPromotion",
]
FEATURES = CATEGORICAL + NUMERIC

data = pd.read_csv("HR-Employee-Attrition.csv")
X = data[FEATURES]
y = (data["Attrition"] == "Yes").astype(int)


def build_pipeline(C=1.0):
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(C=C, max_iter=5000)),
    ])


# 1. Pick the regularisation strength with repeated stratified 5-fold CV.
#    (Repeated CV is used because a single 20% test set is only 294 employees,
#    so one prediction moves accuracy by 0.34% and results depend on luck.)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
search = GridSearchCV(build_pipeline(), {"model__C": [0.03, 0.1, 0.3, 1, 3]},
                      cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X, y)
best_C = search.best_params_["model__C"]

# 2. Report cross-validated performance of the chosen settings.
scores = cross_validate(build_pipeline(best_C), X, y, cv=cv,
                        scoring=["accuracy", "roc_auc", "recall"], n_jobs=-1)
cv_accuracy = scores["test_accuracy"].mean()
print(f"Chosen C: {best_C}")
print(f"Cross-validated accuracy: {cv_accuracy:.4f} (+/- {scores['test_accuracy'].std():.4f})")
print(f"Cross-validated ROC-AUC:  {scores['test_roc_auc'].mean():.4f}")
print(f"Share of leavers caught (recall): {scores['test_recall'].mean():.3f}")

# 3. Also report the same 80/20 split as the original Gradient Boosting script,
#    so the numbers are directly comparable.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
split_model = build_pipeline(best_C).fit(X_train, y_train)
y_pred = split_model.predict(X_test)
split_accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on original 80/20 split: {split_accuracy:.4f}")
print("Classification report:\n", classification_report(y_test, y_pred))

# 4. Train the final model on all the data and save it with its metadata.
final_model = build_pipeline(best_C).fit(X, y)
joblib.dump({
    "model": final_model,
    "features": FEATURES,
    "cv_accuracy": cv_accuracy,
    "split_accuracy": split_accuracy,
}, "attrition_model.pkl")
print("Saved attrition_model.pkl")
