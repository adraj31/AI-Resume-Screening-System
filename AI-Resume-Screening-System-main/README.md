# AI Resume Screening System

A full-stack AI-powered resume screening web application built as a **Final Year B.Tech Computer Science Project**. It predicts a resume's job category, scores it against a job description, and provides detailed AI-powered analysis with improvement suggestions using the **Groq API (LLaMA 3.3 70B)**.

## Features

- **Resume Category Prediction** — Classifies resumes into 25+ job categories using TF-IDF + LinearSVC (85.3% accuracy)
- **Job Match Scoring** — Computes cosine similarity between resume and job description (0–100% with calibrated thresholds)
- **AI-Powered Resume Analysis** — Uses Groq API (LLaMA 3.3 70B) to provide:
  - Professional summary of the candidate
  - Experience level detection (Entry / Mid / Senior / Executive)
  - Overall resume rating (1–10)
  - Strengths and weaknesses analysis
  - Skills identified and recommended skills to add
  - 5 actionable improvement suggestions
  - ATS (Applicant Tracking System) optimization tips
  - Job description match feedback (when JD is provided)
- **Multi-Format Support** — Accepts PDF and DOCX resume uploads with drag-and-drop
- **Privacy First** — Resumes are processed in-memory, never stored on disk
- **Responsive UI** — Mobile-friendly design with sticky navbar, hero sections, and modern styling

## Pages

| Page | Route | Description |
|------|-------|-------------|
| **Home** | `/` | Hero section, How It Works, Upload form, Features, Tech Stack |
| **Results** | `/predict` (POST) | Category prediction, match score, full AI analysis with suggestions |
| **About** | `/about` | Project overview, methodology, model comparison, team members |
| **Contact** | `/contact` | Contact details, contact form, team member directory |

## Project Structure

```
AI-Resume-Screening-System/
├── app.py                   # Flask app: routes for /, /about, /contact, /predict
├── utils.py                 # clean_resume() + predict_category() + match_score()
│                             # (loads the 3 .pkl artifacts once at import)
├── file_extract.py          # PDF (pdfplumber) / DOCX (python-docx) text extraction
├── ai_analysis.py           # Groq API integration for AI-powered resume analysis
├── resume_screening.ipynb   # Training notebook (EDA → cleaning → TF-IDF →
│                             # model comparison → evaluation → JD matching)
├── UpdatedResumeDataSet.csv # Kaggle Resume Dataset (Category, Resume columns)
├── clf.pkl                  # Trained classifier (LinearSVC, best of 3 models)
├── tfidf.pkl                # Fitted TfidfVectorizer (max_features=1500)
├── encoder.pkl              # LabelEncoder for category names
├── templates/
│   ├── base.html             # Navbar, flash messages, footer (shared layout)
│   ├── index.html            # Home page: hero, how it works, upload form, features, tech stack
│   ├── result.html           # Results: category, score, full AI analysis & suggestions
│   ├── about.html            # About: project overview, methodology, team, model comparison
│   └── contact.html          # Contact: details, form, team contacts
├── static/css/style.css      # Complete stylesheet (navy/teal theme, Inter font, responsive)
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| Machine Learning | scikit-learn (LinearSVC, TF-IDF, LabelEncoder) |
| AI Analysis | Groq API with LLaMA 3.3 70B Versatile |
| NLP | NLTK (stopwords, text preprocessing) |
| File Parsing | pdfplumber (PDF), python-docx (DOCX) |
| Frontend | Jinja2 templates, vanilla JavaScript, CSS3 |
| Styling | CSS custom properties, Google Fonts (Inter), responsive grid |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

NLTK stopwords are downloaded automatically on first run.

### 2. Configure Groq API key (optional)

The API key is pre-configured in `ai_analysis.py`. To use your own key, set the environment variable:

```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

### 3. Run the training notebook (optional)

Open `resume_screening.ipynb` and run all cells. It reads `UpdatedResumeDataSet.csv`, trains and compares KNeighborsClassifier, LinearSVC, and RandomForestClassifier, and saves `clf.pkl`, `tfidf.pkl`, and `encoder.pkl`. Re-run this whenever the dataset changes.

### 4. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## How It Works

1. **Upload** — User uploads a resume (PDF/DOCX) and optionally pastes a job description
2. **Text Extraction** — `file_extract.py` extracts raw text from the uploaded file
3. **Preprocessing** — `utils.py` cleans the text (removes URLs, hashtags, mentions, stopwords)
4. **ML Prediction** — TF-IDF vectorizer transforms text, LinearSVC predicts job category
5. **Match Scoring** — If a JD is provided, cosine similarity computes a 0–100% match score
6. **AI Analysis** — `ai_analysis.py` sends the resume to Groq API for detailed feedback
7. **Results** — Everything is displayed on a rich results page with actionable insights

## Model Performance

LinearSVC selected as the best of 3 models on the held-out test split:

| Model | Algorithm | Accuracy | Status |
|-------|-----------|----------|--------|
| **LinearSVC** | Support Vector Classification | **85.3%** | Selected |
| KNN (OvR) | K-Nearest Neighbors | 79.4% | Tested |
| Random Forest | Ensemble Decision Trees | 73.5% | Tested |

Note: the dataset has heavy duplication (only 166 of 962 rows are unique resume/category pairs); the notebook drops duplicates before splitting to avoid train/test leakage.

## Match Score Thresholds

| Score Range | Verdict | Color |
|-------------|---------|-------|
| >= 25% | Strong Match | Green |
| >= 12% | Partial Match | Yellow |
| < 12% | Weak Match | Orange |

Thresholds are calibrated to the model's actual cosine-similarity distribution (same-category median ~28%, cross-category median ~8%).

## Team

| Name | Role |
|------|------|
| Rahul Sharma | Team Lead & ML Engineer |
| Priya Patel | Backend Developer |
| Amit Kumar | Frontend Developer |
| Neha Verma | Data Analyst & Tester |

**Supervisor:** Dr. Rajesh Gupta — Associate Professor, Dept. of Computer Science & Engineering, XYZ University

## Contact

- **Email:** ai.resume.project@gmail.com
- **Phone:** +91 98765 43210
- **Address:** Department of Computer Science & Engineering, XYZ University of Technology, 123 University Road, Tech City, Maharashtra, India - 411001

## License

This project was developed as an academic final year project.
