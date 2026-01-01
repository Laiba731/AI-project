from pprint import pprint
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

# ===============================
# 1. LOAD DATASET
# ===============================
data = pd.read_csv("student_depression_dataset.csv")

# Encode required categorical columns
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Have you ever had suicidal thoughts ?"] = data["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})

# Target column already exists
data["Result"] = data["Depression"].map({1: "Depressed", 0: "Not Depressed"})

# Keep only required columns
data = data[[
    "Gender",
    "Have you ever had suicidal thoughts ?",
    "Result"
]]

# ===============================
# 2. TRAIN TEST SPLIT
# ===============================
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42
)

# ===============================
# 3. ENTROPY
# ===============================
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    ent = 0
    for c in counts:
        p = c / sum(counts)
        ent -= p * math.log2(p)
    return ent


# ===============================
# 4. INFORMATION GAIN
# ===============================
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)

    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = data[data[feature] == v]
        weighted_entropy += (c / sum(counts)) * entropy(subset[target])

    return total_entropy - weighted_entropy


# ===============================
# 5. BUILD TREE (LIMITED DEPTH)
# ===============================
def build_tree(data, features, target, depth=0, max_depth=2):

    # If pure node
    if len(np.unique(data[target])) == 1:
        return data[target].iloc[0]

    # Stop conditions
    if len(features) == 0 or depth == max_depth:
        return data[target].mode()[0]

    # Best feature
    gains = [information_gain(data, f, target) for f in features]
    best_feature = features[np.argmax(gains)]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = build_tree(
            subset, remaining_features, target, depth + 1, max_depth
        )

    return tree


# ===============================
# 6. TRAIN MODEL
# ===============================
features = [
    "Gender",
    "Have you ever had suicidal thoughts ?"
]

decision_tree = build_tree(train_data, features, "Result")

print("\nDECISION TREE STRUCTURE:\n")
pprint(decision_tree)


# ===============================
# 7. PREDICTION FUNCTION
# ===============================
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample[feature]

    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        return None


# ===============================
# 8. TEST MODEL
# ===============================
y_true = test_data["Result"].values
y_pred = []

for _, row in test_data.iterrows():
    y_pred.append(predict(decision_tree, row))


# ===============================
# 9. ACCURACY
# ===============================
correct = sum(y_true[i] == y_pred[i] for i in range(len(y_true)))
accuracy = correct / len(y_true)

print("\nMODEL ACCURACY:", accuracy)