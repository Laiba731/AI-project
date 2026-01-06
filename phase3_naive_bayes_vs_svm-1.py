# =========================================================
# PHASE 3: NAÏVE BAYES vs SVM (SCIKIT-LEARN)
# Student Mental Health Dataset
# Target Variable: Depression
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv("mental_health_students.csv")

print("Dataset Loaded")
print("Shape:", df.shape)

# ---------------------------------------------------------
# 2. ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Encoding Completed")

# ---------------------------------------------------------
# 3. FEATURES AND TARGET
# ---------------------------------------------------------
X = df.drop("Depression", axis=1)
y = df["Depression"]

# ---------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 5. FEATURE SCALING
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 6. NAÏVE BAYES
# ---------------------------------------------------------
start = time.time()

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

nb_time = time.time() - start

y_pred_nb = nb.predict(X_test_scaled)
y_prob_nb = nb.predict_proba(X_test_scaled)[:, 1]

nb_acc = accuracy_score(y_test, y_pred_nb)
nb_auc = roc_auc_score(y_test, y_prob_nb)

print("\nNaïve Bayes Results")
print("Accuracy:", nb_acc)
print("ROC-AUC:", nb_auc)
print("Training Time:", nb_time)
