import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset (correct format)
data = pd.read_csv("spam.csv", sep="\t", encoding="latin-1", header=None)

data = data[[0,1]]
data.columns = ["label","message"]

# Convert labels to numbers
data["label"] = data["label"].map({"ham":0, "spam":1})

# Remove empty rows
data = data.dropna()

# Convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(data["message"])
y = data["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Test message
msg = input("Enter message: ")
msg_vector = cv.transform([msg])
prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")