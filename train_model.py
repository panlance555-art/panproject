
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# 1. LOAD AND PREPARE THE DATASET
# =====================================

dataset_path = "dataset/SMSSpamCollection"

data = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

data = data.drop_duplicates()
data = data.dropna(subset=["label", "message"])

data["message"] = data["message"].astype(str).str.strip()
data["label"] = data["label"].str.strip().str.lower()

data = data[data["label"].isin(["ham", "spam"])]

X = data["message"]
y = data["label"]

# =====================================
# 2. SPLIT THE DATASET
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("SMS SPAM DETECTION SYSTEM")
print("=========================")
print("Total messages:", len(data))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

# =====================================
# 3. DEFINE THE MACHINE LEARNING MODELS
# =====================================

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}

results = {}
best_model = None
best_model_name = None
best_f1 = -1

# =====================================
# 4. TRAIN AND EVALUATE EACH MODEL
# =====================================

for name, algorithm in models.items():

    print("\nTraining:", name)

    # Create TF-IDF and model pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", algorithm)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict testing messages
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print("\nResults for", name)
    print("-------------------------")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Spam Precision:", round(precision * 100, 2), "%")
    print("Spam Recall:", round(recall * 100, 2), "%")
    print("Spam F1-Score:", round(f1 * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Select the best model using spam F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

# =====================================
# 5. DISPLAY MODEL COMPARISON
# =====================================

print("\nMODEL COMPARISON")
print("=========================")

comparison = pd.DataFrame(results).T
print((comparison * 100).round(2))

# =====================================
# 6. SAVE THE BEST MODEL
# =====================================

joblib.dump(best_model, "model/spam_detector.joblib")

print("\nBEST MODEL:", best_model_name)
print("Best Spam F1-Score:", round(best_f1 * 100, 2), "%")
print("Model successfully saved!")
print("\nTRAINING COMPLETED!")
