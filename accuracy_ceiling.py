"""How high can accuracy honestly go on this dataset, and how are '95%+' results produced?

Reproduces the tables in the README section "Why not 95%+ accuracy?".
Needs the packages in requirements-dev.txt. Takes a few minutes to run.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from catboost import CatBoostClassifier

df = pd.read_csv("HR-Employee-Attrition.csv").drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
y = (df.pop("Attrition") == "Yes").astype(int)

# --- Feature engineering commonly used on this dataset
fe = df.copy()
fe["IncomePerLevel"] = fe.MonthlyIncome / fe.JobLevel
fe["YearsPerCompany"] = fe.TotalWorkingYears / (fe.NumCompaniesWorked + 1)
fe["TenureRatio"] = fe.YearsAtCompany / (fe.TotalWorkingYears + 1)
fe["RoleRatio"] = fe.YearsInCurrentRole / (fe.YearsAtCompany + 1)
fe["TotalSatisfaction"] = fe[["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
                              "WorkLifeBalance", "JobInvolvement"]].sum(axis=1)
fe["LogIncome"] = np.log(fe.MonthlyIncome)
fe["Young"] = (fe.Age < 25).astype(int)

def lr(X, C=0.3):
    cat = X.select_dtypes(exclude="number").columns.tolist(); num = [c for c in X if c not in cat]
    return make_pipeline(ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat), ("n", StandardScaler(), num)]),
                         LogisticRegression(C=C, max_iter=5000))

class Tuned(BaseEstimator, ClassifierMixin):
    """Pick the decision threshold that maximises accuracy using inner CV (no test leakage)."""
    def __init__(self, est): self.est = est
    def fit(self, X, y):
        p = cross_val_predict(clone(self.est), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        ts = np.linspace(0.2, 0.8, 61)
        self.t_ = ts[np.argmax([((p >= t) == y).mean() for t in ts])]
        self.m_ = clone(self.est).fit(X, y); self.classes_ = np.array([0, 1]); return self
    def predict(self, X): return (self.m_.predict_proba(X)[:, 1] >= self.t_).astype(int)

cat_cols = df.select_dtypes(exclude="number").columns.tolist()
def onehot(X):
    num = [c for c in X if c not in cat_cols]
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("n", StandardScaler(), num)])
stack = StackingClassifier([
    ("lr", make_pipeline(onehot(fe), LogisticRegression(C=0.3, max_iter=5000))),
    ("svm", make_pipeline(onehot(fe), SVC(kernel="linear", C=0.05, probability=True))),
    ("cb", make_pipeline(ColumnTransformer([("c", OrdinalEncoder(), cat_cols)], remainder="passthrough"),
                         CatBoostClassifier(iterations=500, depth=4, verbose=0, random_seed=42, thread_count=4))),
], final_estimator=LogisticRegression(), cv=5)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
print(f"Always guess 'stays':                       {1 - y.mean():.4f}")
print("HONEST (cross-validated):")
for name, m, X in [("LR, all features", lr(df), df),
                   ("LR + engineered features", lr(fe), fe),
                   ("LR + engineered + tuned threshold", Tuned(lr(fe)), fe),
                   ("Stacking LR+SVM+CatBoost + engineered", stack, fe)]:
    print(f"  {name:40s} {cross_val_score(m, X, y, cv=cv, n_jobs=-1).mean():.4f}", flush=True)

print("LEAKY (how inflated numbers are produced):")
Xo = pd.get_dummies(df, dtype=float)
# 1) Oversample leavers to 50/50 BEFORE splitting -> copies of the same employee end up in train AND test
rng = np.random.default_rng(0)
idx = np.r_[np.where(y == 0)[0], rng.choice(np.where(y == 1)[0], (y == 0).sum())]
Xa, ya = Xo.iloc[idx], y.iloc[idx]
Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
print(f"  Oversample before split + RandomForest     {rf.score(Xte, yte):.4f}")
# 2) Score on the training data
print(f"  Accuracy on the training data itself       {RandomForestClassifier(random_state=42).fit(Xo, y).score(Xo, y):.4f}")
# 3) Same model, honest evaluation, for contrast
print(f"  Same RandomForest, honest CV               {cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42), Xo, y, cv=cv, n_jobs=-1).mean():.4f}")
