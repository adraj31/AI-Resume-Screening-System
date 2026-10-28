"""
AI Resume Screening System — Flask app.

Loads the artifacts produced by resume_screening.ipynb (clf.pkl, tfidf.pkl,
encoder.pkl) once at startup via utils.py, then exposes:
  GET  /         upload form (resume file + optional job description)
  POST /predict  extract -> clean -> predict category -> optional match score
                 + Groq AI-powered detailed analysis & suggestions
"""
import os

from flask import Flask, render_template, request, redirect, url_for, flash

from utils import predict_category, match_score
from file_extract import extract_text_from_file, allowed_file
from ai_analysis import analyze_resume

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Match-score verdict thresholds (0-100). Calibrated against this model's actual
# cosine-similarity distribution: same-category resume pairs score ~6-59%
# (median ~28%), cross-category pairs ~1-35% (median ~8%) -- a naive 70/40
# split would mislabel nearly every real match as "Weak".
STRONG_THRESHOLD = 25
PARTIAL_THRESHOLD = 12


def score_verdict(score):
    """Map a 0-100 match score to a (label, level) pair for display."""
    if score >= STRONG_THRESHOLD:
        return 'Strong match', 'high'
    if score >= PARTIAL_THRESHOLD:
        return 'Partial match', 'medium'
    return 'Weak match', 'low'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('resume_file')
    jd_text = request.form.get('jd_text', '').strip()

    if not uploaded_file or not uploaded_file.filename:
        flash('Please upload a resume file (PDF or DOCX).')
        return redirect(url_for('index'))

    if not allowed_file(uploaded_file.filename):
        flash('Unsupported file format. Please upload a .pdf or .docx file.')
        return redirect(url_for('index'))

    try:
        resume_text = extract_text_from_file(uploaded_file)
    except Exception as e:
        flash(f'Could not read the uploaded file: {e}')
        return redirect(url_for('index'))

    if not resume_text.strip():
        flash('No readable text found in the uploaded resume.')
        return redirect(url_for('index'))

    category = predict_category(resume_text)
    result = {
        'filename': uploaded_file.filename,
        'category': category,
    }

    if jd_text:
        score = match_score(resume_text, jd_text)
        verdict, level = score_verdict(score)
        result['score'] = score
        result['verdict'] = verdict
        result['level'] = level

    # AI-powered detailed analysis via Groq
    analysis = analyze_resume(resume_text, category, jd_text if jd_text else None)
    result['analysis'] = analysis

    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
