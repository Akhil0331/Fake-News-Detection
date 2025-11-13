"""
Helper functions to load the trained model and predict single text.
"""
import joblib
from utils import clean_text


MODEL_PATH = 'model.joblib'




def load_model(path=MODEL_PATH):
model = joblib.load(path)
return model




def predict_text(text: str, model=None):
if model is None:
model = load_model()
cleaned = clean_text(text)
pred = model.predict([cleaned])[0]
prob = None
if hasattr(model, 'predict_proba'):
prob = model.predict_proba([cleaned])[0].max()
return int(pred), float(prob) if prob is not None else None
