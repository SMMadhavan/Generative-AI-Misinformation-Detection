import numpy
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


print("Working Perfect...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print(fake.shape, true.shape)

fake["label"] = 1   # misinformation
true["label"] = 0   # authentic

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

import re

def clean_text(text):  #Data CLeaning
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["clean_text"] = data["text"].apply(clean_text)

X = data["clean_text"]
y = data["label"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Create train DataFrame
train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

# Create test DataFrame
test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})

# Save to data folder
train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

print("Train and test data saved inside data folder")


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import numpy as np

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_misinfo_indices = np.argsort(coefficients)[-20:]
top_authentic_indices = np.argsort(coefficients)[:20]

print("Top words indicating Misinformation:")
for idx in reversed(top_misinfo_indices):
    print(feature_names[idx], coefficients[idx])

print("\nTop words indicating Authentic content:")
for idx in top_authentic_indices:
    print(feature_names[idx], coefficients[idx])

data["predicted_label"] = model.predict(
    vectorizer.transform(data["clean_text"])
)

plt.figure(figsize=(6,4))
sns.countplot(x="predicted_label", data=data)
plt.xticks([0, 1], ["Authentic", "Misinformation"])
plt.title("Overall Misinformation vs Authentic Content")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.show()

misinfo_count = data["predicted_label"].value_counts()
print(misinfo_count)
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data["year"] = data["date"].dt.year

yearly_trend = data.groupby("year")["predicted_label"].mean().dropna()

plt.figure(figsize=(7,4))
yearly_trend.plot(kind="line", marker="o")
plt.title("Year-wise Misinformation Trend")
plt.xlabel("Year")
plt.ylabel("Average Misinformation Probability")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
sns.barplot(x=yearly_trend.index, y=yearly_trend.values)
plt.title("Average Misinformation by Year")
plt.xlabel("Year")
plt.ylabel("Misinformation Rate")
plt.show()

trend = data.groupby("year")["predicted_label"].mean()
print(trend)

def get_misinfo_probability(text):
    text_vec = vectorizer.transform([text])
    prob = model.predict_proba(text_vec)[0][1]
    return prob



sample_texts = [
    "NASA has confirmed that the Earth will experience six days of complete darkness next month due to a planetary alignment."
]

sample_vec = vectorizer.transform(sample_texts)

prediction = model.predict(sample_vec)
probability = model.predict_proba(sample_vec)

print("Prediction:", "Misinformation" if prediction[0] == 1 else "Authentic")

print("Misinformation probability:", probability[0][1])

#Model Performs prototype fuctioality and is enhanced progressively

