import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Loading the dataset
datasetpath = os.path.join(os.path.dirname(__file__), 'training_dataset.csv')
data = pd.read_csv(datasetpath ,delimiter=',', header=None)
data.columns = ['column1', 'column2', 'column3', 'column4']

X = data['column4']  # contains the text data
y = data['column3']  # contains labels 

X.fillna('', inplace=True) # handle missing value

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data using the Bag-of-Words approach
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
vocabulary = vectorizer.get_feature_names_out()

# Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

myclassifier = {
    'model':classifier,
    'vocabulary':vocabulary
}

outputmodelpath = os.path.join(os.path.dirname(__file__), 'sentiment_analysis.joblib')
joblib.dump(myclassifier, outputmodelpath)