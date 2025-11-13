import re
import nltk
from nltk.corpus import stopwords


# Ensure NLTK resources are downloaded the first time
try:
_ = stopwords.words('english')
except LookupError:
nltk.download('stopwords')


STOPWORDS = set(stopwords.words('english'))




def clean_text(text: str) -> str:
"""Basic text cleaning: lowercasing, remove urls, non-alpha chars, extra spaces, remove stopwords.


This is intentionally simple and fast for student projects.
"""
if not isinstance(text, str):
return ''
text = text.lower()
# remove URLs
text = re.sub(r'http\S+|www\.\S+', ' ', text)
# remove non-alphanumeric characters (keep spaces)
text = re.sub(r'[^a-z0-9\s]', ' ', text)
# collapse whitespace
text = re.sub(r'\s+', ' ', text).strip()
# remove stopwords
tokens = [w for w in text.split() if w not in STOPWORDS]
return ' '.join(tokens)
