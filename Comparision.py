import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# Advanced ensemble models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("student_depression_dataset.csv")

# Binary target (already available)
df["Result"] = df["Depression"]   # 1 = Depressed, 0 = Not Depressed

# Select features (drop target)
X = df.drop(columns=["Depression", "Result"])
y = df["Result"]

# Convert categorical features to numeric
X = pd.get_dummies(X, drop_first=True)


# =========================
# 2. TRAIN–TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 3. MODELS
# =========================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}


# =========================
# 4. EVALUATION
# =========================

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ])


# =========================
# 5. DISPLAY RESULTS
# =========================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
)

print("\nEVALUATION METRICS OF ALL MODELS\n")
print(results_df.round(3))