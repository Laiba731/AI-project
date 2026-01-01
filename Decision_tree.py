import pandas as pd
import numpy as np
from collections import Counter
from graphviz import Digraph

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv(r"C:\Users\786\OneDrive\Desktop\phase1_ai\student_depression_dataset.csv")

# Encode categorical variables (binary)
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})

# Target variable
y = df["Depression"]

# Selected features (binary)
X = df[[
    "Gender",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]]

# =========================
# 2. ENTROPY
# =========================

def entropy(y):
    counts = Counter(y)
    total = len(y)
    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * np.log2(p)
    return ent

# =========================
# 3. INFORMATION GAIN
# =========================

def info_gain(X_col, y):
    parent_entropy = entropy(y)
    values = np.unique(X_col)

    weighted_entropy = 0
    for v in values:
        subset_y = y[X_col == v]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    return parent_entropy - weighted_entropy

# =========================
# 4. BUILD TREE
# =========================

class MyDecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]

        gains = {col: info_gain(X[col], y) for col in X.columns}
        best_feature = max(gains, key=gains.get)

        tree = {best_feature: {}}

        for val in np.unique(X[best_feature]):
            sub_X = X[X[best_feature] == val]
            sub_y = y[X[best_feature] == val]
            tree[best_feature][val] = self.fit(sub_X, sub_y, depth + 1)

        return tree

# =========================
# 5. TRAIN TREE
# =========================

model = MyDecisionTree(max_depth=2)
tree = model.fit(X, y)

# =========================
# 6. GRAPHVIZ VISUALIZATION
# =========================

dot = Digraph(
    format="png",
    node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#E8F6F3"}
)

node_id = 0

def draw(tree, parent=None, label=""):
    global node_id
    current = str(node_id)
    node_id += 1

    if isinstance(tree, dict):
        feature = list(tree.keys())[0]
        dot.node(current, feature)

        if parent:
            dot.edge(parent, current, label)

        for value, subtree in tree[feature].items():
            draw(subtree, current, str(value))
    else:
        result = "Depression" if tree == 1 else "No Depression"
        dot.node(current, result, shape="ellipse", fillcolor="#FDEDEC")
        if parent:
            dot.edge(parent, current, label)

draw(tree)
dot.render("Student_Depression_Decision_Tree", cleanup=True)

print("✅ Student_Depression_Decision_Tree.png saved successfully")