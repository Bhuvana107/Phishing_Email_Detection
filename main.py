
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
data = pd.read_csv("emails.csv")

X = data["email_text"].fillna("")
y = data["label"]

# Step 2: Extract email features
suspicious_words = [
    "urgent", "verify", "password", "prize",
    "claim", "free", "blocked", "reward",
    "suspended", "lottery"
]

def extract_features(email):
    email = email.lower()

    url_count = len(re.findall(r"https?://\S+|www\.\S+", email))
    keyword_count = sum(word in email for word in suspicious_words)
    exclamation_count = email.count("!")
    email_length = len(email)

    return [
        url_count,
        keyword_count,
        exclamation_count,
        email_length
    ]

features = [extract_features(email) for email in X]

# Step 3: Split dataset
X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
    X, y, features,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Step 4: Convert text into numerical features
vectorizer = TfidfVectorizer()
X_train_text = vectorizer.fit_transform(X_train)
X_test_text = vectorizer.transform(X_test)

X_train_final = hstack([X_train_text, csr_matrix(f_train)])
X_test_final = hstack([X_test_text, csr_matrix(f_test)])

# Step 5: Train Machine Learning model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

# Step 6: Predict results
predictions = model.predict(X_test_final)

# Step 7: Display accuracy
accuracy = accuracy_score(y_test, predictions)

print("PHISHING EMAIL DETECTION MODEL")
print("--------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Step 8: Confusion Matrix
cm = confusion_matrix(
    y_test, predictions,
    labels=["Phishing", "Safe"]
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Safe"],
    yticklabels=["Phishing", "Safe"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.show()
print("\n--- Test Your Own Email ---")

new_email = input("Enter an email message: ")

new_features = extract_features(new_email)

new_text = vectorizer.transform([new_email])

new_final = hstack([
    new_text,
    csr_matrix([new_features])
])

result = model.predict(new_final)

print("Email Classification:", result[0])