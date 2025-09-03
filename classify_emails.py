import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# -----------------------------------------------------
# 1. Load the CSV dataset
# -----------------------------------------------------
data = pd.read_csv("Sample_Support_Emails_Dataset.csv")  # keep file in same folder

# -----------------------------------------------------
# 2. Define helper functions
# -----------------------------------------------------
priority_keywords = ['urgent', 'immediate', 'help', 'blocked', 'error']

def detect_priority(text):
    """Return 1 if any priority keyword appears, else 0."""
    text = text.lower()
    return int(any(word in text for word in priority_keywords))

def categorize_email(text):
    """Basic keyword-based categorization for small dataset."""
    text = text.lower()
    if 'account' in text or 'login' in text or 'password' in text:
        return 'Account Issue'
    elif 'billing' in text or 'payment' in text or 'pricing' in text:
        return 'Billing Issue'
    elif 'subscription' in text:
        return 'Subscription'
    elif 'api' in text or 'integration' in text:
        return 'Technical/Integration'
    else:
        return 'General Query'

# -----------------------------------------------------
# 3. Add derived columns
# -----------------------------------------------------
combined_text = data['subject'].fillna('') + " " + data['body'].fillna('')
data['priority'] = combined_text.apply(detect_priority)
data['category'] = combined_text.apply(categorize_email)

# -----------------------------------------------------
# 4. Split for training / testing
# -----------------------------------------------------
X = combined_text
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------
# 5. Build TF-IDF + Naive Bayes pipeline
# -----------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

# -----------------------------------------------------
# 6. Train & Evaluate
# -----------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# 7. Save results for reference
# -----------------------------------------------------
output = data[['sender', 'subject', 'priority', 'category']]
output.to_csv("classified_emails_output.csv", index=False)
print("✅ Results saved to classified_emails_output.csv")
