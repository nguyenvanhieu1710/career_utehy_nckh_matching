# Career UTEHY NCKH - CV Job Matching Service

An AI-powered service that matches CVs to job descriptions using semantic similarity and machine learning for the UTEHY Career Platform.

## 🎯 Overview

This service provides intelligent CV-to-job matching capabilities for the Career UTEHY NCKH platform, helping candidates find the most suitable job opportunities based on their skills and experience.

## ⚙️ Installation & Setup

1️⃣ Clone the repository

```bash
git clone https://github.com/nguyenvanhieu1710/career-utehy-nckh-matching.git
cd career-utehy-nckh-matching
```

2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

4️⃣ Configure environment variables

```bash
cp .env.example .env
```

## ▶️ Run the Application

```bash
python -m app.main
# Or: uvicorn app.main:app --reload
```

## 📖 Usage

Access the API documentation and test endpoints:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

## 🚀 API Endpoints

### CV-Job Matching

- `POST /api/v1/match/cv-file` - Upload CV PDF file for matching
- `POST /api/v1/match/cv-json` - Submit CV data in JSON format

### Health Check

- `GET /api/v1/health/` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health with database status

## 🎯 Features

- **Dual CV Input Support:** PDF files and JSON structured data
- **MongoDB Integration:** Connects to Career UTEHY NCKH database
- **Intelligent Matching:** Semantic similarity + skill analysis
- **Skill Recommendations:** Suggests improvements for better matches
- **Compatibility Scoring:** Percentage-based match scores
- **Production Ready:** Error handling, logging, health checks
