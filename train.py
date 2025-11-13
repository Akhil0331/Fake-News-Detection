"""
Train a Fake News classifier and save the trained model to disk (model.joblib).


Expected dataset format (CSV): columns `title`, `text`, `label` (label values: 'FAKE'/'REAL' or 0/1)
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from utils import clean_text


DATA_PATH = os.path.join('data', 'train.csv')
MODEL_PATH = 'model.joblib'




def load_data(path=DATA_PATH):
df = pd.read_csv(path)
# Try to combine title and text if title exists
if 'title' in df.columns:
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
else:
df['content'] = df['text'].fillna('')


# normalize labels to 0/1
if df['label'].dtype == object:
df['label'] = df['label'].map(lambda x: 1 if str(x).strip().lower() in ('fake','1','true','yes') else 0)
else:
df['label'] = df['label'].astype(int)
return df[['content','label']]




def preprocess_series(series: pd.Series):
return series.fillna('').astype(str).apply(clean_text)




def main():
if not os.path.exists(DATA_PATH):
raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please place your train.csv at data/train.csv")


df = load_data()
X = preprocess_series(df['content'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)


pipeline = Pipeline([
('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
('clf', LogisticRegression(max_iter=1000))
])


print('Training model...')
pipeline.fit(X_train, y_train)


preds = pipeline.predict(X_test)
print('\n== Evaluation on test set ==')
print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print('Confusion matrix:\n', confusion_matrix(y_test, preds))


print(f'Saving model to {MODEL_PATH}...')
joblib.dump(pipeline, MODEL_PATH)
print('Done.')




if __name__ == '__main__':
main()
