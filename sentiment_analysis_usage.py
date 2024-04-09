from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'sentiment_analysis.joblib')
model = joblib.load(model_path)

vectorizer = CountVectorizer(vocabulary=model['vocabulary'])

classifier = model['model']
text = "too loose"

mytext = vectorizer.transform([text])
print(classifier.predict(mytext)[0])