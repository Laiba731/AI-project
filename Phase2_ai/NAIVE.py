import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("student_depression_dataset.csv")  # <-- apna file name yahan rakho

# =========================
# 2. TARGET VARIABLE
# =========================
# Depression already binary (0 = No, 1 = Yes)
y = df["Depression"]

# Drop ID & target column
df = df.drop(columns=["id", "Depression"])

# =========================
# 3. CONVERT ROWS TO TEXT
# =========================
def row_to_text(row):
    return " ".join([f"{col}_{row[col]}" for col in df.columns])

df["text"] = df.apply(row_to_text, axis=1)

X = df["text"]

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. TEXT VECTORIZATION
# =========================
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# =========================
# 6. MULTINOMIAL NAÏVE BAYES (FROM SCRATCH)
# =========================
class MultinomialNB_Scratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = {}
        self.likelihood = {}

        for c in self.classes:
            X_c = X[y == c]
            self.prior[c] = X_c.shape[0] / X.shape[0]

            word_counts = X_c.sum(axis=0) + 1  # Laplace smoothing
            self.likelihood[c] = word_counts / word_counts.sum()

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                log_prior = np.log(self.prior[c])
                log_likelihood = np.sum(x * np.log(self.likelihood[c]))
                posteriors.append(log_prior + log_likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# =========================
# 7. TRAIN & PREDICT (NB)
# =========================
nb = MultinomialNB_Scratch()
nb.fit(X_train_vec, y_train)
nb_preds = nb.predict(X_test_vec)

# =========================
# 8. EVALUATION (NB)
# =========================
print("NAÏVE BAYES RESULTS")
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))
print("Precision:", precision_score(y_test, nb_preds))
print("Recall:", recall_score(y_test, nb_preds))
print("F1-score:", f1_score(y_test, nb_preds))

# =========================
# 9. DECISION TREE COMPARISON
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_vec, y_train)
dt_preds = dt.predict(X_test_vec)

print("\nDECISION TREE RESULTS")
print("Precision:", precision_score(y_test, dt_preds))
print("Recall:", recall_score(y_test, dt_preds))
print("F1-score:", f1_score(y_test, dt_preds))
