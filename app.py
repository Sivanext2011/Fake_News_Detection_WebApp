from flask import Flask, render_template
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

app = Flask(__name__)

# Load datasets
true = pd.read_csv('data/True.csv')
fake = pd.read_csv('data/Fake.csv')

# Labeling the data
true['label'] = 1
fake['label'] = 0

# Merging datasets
news_data = pd.concat([fake, true], axis=0)

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n', '', text)
    return text

# Apply preprocessing
news_data['text'] = news_data['text'].apply(wordopt)

# Split data into features and labels
X = news_data['text']
Y = news_data['label']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Vectorization
vectorization = TfidfVectorizer()
XV_train = vectorization.fit_transform(X_train)
XV_test = vectorization.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(XV_train, Y_train)

@app.route('/')
def index():
    # Make predictions on the test set
    predictions = {}
    for name, model in models.items():
        preds = model.predict(XV_test)
        predictions[name] = [output_label(p) for p in preds]
    
    # Render the result page with the predictions
    return render_template('result.html', predictions=predictions)

def output_label(n):
    return "Fake News" if n == 0 else "True News"

if __name__ == '__main__':
    app.run(debug=True)
