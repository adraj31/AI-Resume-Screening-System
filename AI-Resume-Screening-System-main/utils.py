"""
Shared preprocessing/inference logic for the AI Resume Screening System.

clean_resume() is copied verbatim from the training notebook
(resume_screening.ipynb, Phase 3) so the Flask app applies the exact same
text normalization the model was trained on.
"""
import os
import re
import pickle

import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOPWORDS = set(stopwords.words('english'))


def clean_resume(text):
    """Clean raw resume/job-description text for TF-IDF vectorization.

    Steps: remove URLs, RT/cc tokens, hashtags, mentions, punctuation,
    non-ASCII characters, and extra whitespace; lowercase; strip English
    stopwords. Must stay identical to the notebook's version.
    """
    text = str(text)

    text = re.sub(r'http\S+\s*', ' ', text)          # URLs
    text = re.sub(r'RT|cc', ' ', text)               # RT / cc tokens
    text = re.sub(r'#\S+', ' ', text)                # hashtags
    text = re.sub(r'@\S+', ' ', text)                # mentions
    text = re.sub(r'[^\x00-\x7f]', ' ', text)        # non-ASCII characters
    text = re.sub(r'[%s]' % re.escape(
        '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), ' ', text)  # punctuation
    text = re.sub(r'\s+', ' ', text)                 # extra whitespace

    text = text.lower().strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]

    return ' '.join(tokens)


def _load_artifact(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {filename}. Run resume_screening.ipynb first to generate it."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


clf = _load_artifact('clf.pkl')
tfidf = _load_artifact('tfidf.pkl')
encoder = _load_artifact('encoder.pkl')


def predict_category(raw_resume_text):
    """Raw resume text -> predicted category name."""
    cleaned = clean_resume(raw_resume_text)
    features = tfidf.transform([cleaned])
    predicted_label = clf.predict(features)[0]
    return encoder.inverse_transform([predicted_label])[0]


def match_score(resume_text, jd_text):
    """Cosine-similarity match between a resume and a job description, as 0-100."""
    cleaned_resume_text = clean_resume(resume_text)
    cleaned_jd_text = clean_resume(jd_text)

    vectors = tfidf.transform([cleaned_resume_text, cleaned_jd_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return round(similarity * 100, 2)
