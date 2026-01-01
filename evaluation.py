import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("student_depression_dataset.csv")

# Encode required categorical features
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})

# Create target
df["Result"] = df["Depression"].map({1: "Depressed", 0: "Not Depressed"})

# Select features
X = df[[
    "Gender",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]]
y = df["Result"]


# =========================
# 2. TRAIN–TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 3. ENTROPY
# =========================

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    ent = 0
    for c in counts:
        p = c / sum(counts)
        ent -= p * np.log2(p)
    return ent


# =========================
# 4. INFORMATION GAIN
# =========================

def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)

    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset_y = y[X[feature] == v]
        weighted_entropy += (c / sum(counts)) * entropy(subset_y)

    return total_entropy - weighted_entropy


# =========================
# 5. BEST FEATURE
# =========================

def best_feature(X, y):
    gains = {f: information_gain(X, y, f) for f in X.columns}
    return max(gains, key=gains.get)


# =========================
# 6. BUILD DECISION TREE (DEPTH LIMITED)
# =========================

def build_tree(X, y, depth=0, max_depth=2):
    if len(np.unique(y)) == 1:
        return y.iloc[0]

    if X.shape[1] == 0 or depth == max_depth:
        return y.mode()[0]

    best = best_feature(X, y)
    tree = {best: {}}

    for value in np.unique(X[best]):
        sub_X = X[X[best] == value].drop(columns=best)
        sub_y = y[X[best] == value]
        tree[best][value] = build_tree(sub_X, sub_y, depth + 1, max_depth)

    return tree


# =========================
# 7. TRAIN TREE
# =========================

decision_tree = build_tree(X_train, y_train)

print("\nDECISION TREE STRUCTURE:\n")
print(decision_tree)


# =========================
# 8. PREDICTION FUNCTION
# =========================

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample[feature]

    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        return y_train.mode()[0]


# =========================
# 9. GET PREDICTIONS
# =========================

y_pred = []
for _, row in X_test.iterrows():
    y_pred.append(predict(decision_tree, row))


# =========================
# 10. CONFUSION MATRIX
# =========================

TP = FP = TN = FN = 0

for actual, predicted in zip(y_test, y_pred):
    if actual == "Depressed" and predicted == "Depressed":
        TP += 1
    elif actual == "Not Depressed" and predicted == "Depressed":
        FP += 1
    elif actual == "Not Depressed" and predicted == "Not Depressed":
        TN += 1
    elif actual == "Depressed" and predicted == "Not Depressed":
        FN += 1


# =========================
# 11. EVALUATION METRICS
# =========================

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = (2 * precision * recall /
            (precision + recall)) if (precision + recall) != 0 else 0


# =========================
# 12. PRINT RESULTS
# =========================

print("\nMODEL EVALUATION METRICS")
print("------------------------")
print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1-score :", round(f1_score, 3))