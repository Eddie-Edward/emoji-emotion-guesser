import pandas as pd
df = pd.read_csv("emotions.csv")
print(df.head())

import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download("punkt")  # Tokenizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X, df["emotion"])

sentence = "I'm feeling amazing!"
vector = vectorizer.transform([sentence])
prediction = model.predict(vector)
print("Predicted emotion:", prediction[0])

emojis = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠"
}

print("Emoji:", emojis.get(prediction[0], "🤔"))
