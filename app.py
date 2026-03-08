import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Load dataset
data = pd.read_csv("spam.csv", sep="\t", encoding="latin-1", header=None)
data = data[[0,1]]
data.columns = ["label", "message"]
data["label"] = data["label"].map({"ham":0, "spam":1})
data = data.dropna()

# Train model
cv = CountVectorizer()
X = cv.fit_transform(data["message"])
y = data["label"]

model = MultinomialNB()
model.fit(X, y)

# Flask app
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Spam Email Detector</title>
</head>
<body>
<h2>Spam Email Detector</h2>

<form method="post">
<textarea name="message" rows="5" cols="40" placeholder="Enter message here"></textarea><br><br>
<button type="submit">Check</button>
</form>

{% if result %}
<h3>Result: {{ result }}</h3>
{% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        msg = request.form["message"]
        msg_vector = cv.transform([msg])
        prediction = model.predict(msg_vector)
        result = "Spam Message" if prediction[0] == 1 else "Not Spam"

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
