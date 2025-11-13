from flask import Flask, render_template, request, jsonify
from model import load_model, predict_text
import os


app = Flask(__name__)
model = None


@app.before_first_request
def load():
global model
if os.path.exists('model.joblib'):
model = load_model('model.joblib')
else:
model = None




@app.route('/')
def index():
return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
data = request.form or request.json
text = data.get('text', '')
if not text:
return jsonify({'error': 'No text provided'}), 400


if model is None:
return jsonify({'error': 'Model not found. Train the model first (run train.py).'}), 500


pred, prob = predict_text(text, model=model)
label = 'FAKE' if pred == 1 else 'REAL'
return jsonify({'label': label, 'probability': prob})




if __name__ == '__main__':
app.run(debug=True)
