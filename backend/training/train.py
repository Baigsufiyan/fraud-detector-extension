import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("../../data/dataset.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2
)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model
pickle.dump(model, open("../model/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("../model/vectorizer.pkl", "wb"))