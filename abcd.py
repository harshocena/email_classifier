import re
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -------------------------------
# Text preprocessing
# -------------------------------
def process_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words
        and len(word) > 2
    ]

    return " ".join(tokens)


# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_excel("tricky_email_dataset.xlsx")

df.columns = df.columns.str.strip().str.lower()

print("\nColumns found:")
print(df.columns.tolist())

# -------------------------------
# Detect email text column
# -------------------------------
possible_columns = [
    "body",
    "message",
    "content",
    "text",
    "email_body",
    "mail",
    "description"
]

text_column = None

for col in possible_columns:
    if col in df.columns:
        text_column = col
        break

if text_column is None:
    raise ValueError(
        f"No email text column found.\n"
        f"Available columns are: {df.columns.tolist()}"
    )

print(f"\nUsing '{text_column}' as email text column.")

# -------------------------------
# Check label column
# -------------------------------
if "label" not in df.columns:
    raise ValueError(
        f"'label' column not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

# -------------------------------
# Clean data
# -------------------------------
df[text_column] = df[text_column].fillna("").astype(str)

df["body_clean"] = (
    df[text_column]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df.drop_duplicates(subset=["body_clean"])

df["clean_text"] = df[text_column].apply(process_text)

X = df["clean_text"]
y = df["label"]

# -------------------------------
# Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Model
# -------------------------------
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        )
    )
])

# -------------------------------
# Train
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Cross Validation
# -------------------------------
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(scores)

print("\nMean CV Accuracy:")
print(scores.mean())

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, "email_classifier.pkl")

print("\nModel saved as email_classifier.pkl")