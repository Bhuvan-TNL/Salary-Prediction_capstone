# 💼 Salary Prediction Capstone

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready machine learning application that predicts expected Cost To Company (CTC) salary packages based on candidate attributes. Built with Random Forest regression and deployed as an interactive Flask web application.

**[🔗 Live Demo](https://salary-prediction-capstone.onrender.com)** | **[📂 Repository](https://github.com/Bhuvan-TNL/Salary-Prediction_capstone)** | **[📊 GitHub](https://github.com/Bhuvan-TNL)**

---

## 📋 Project Overview

This project combines machine learning model development with web application design to create a practical salary prediction tool. It demonstrates the complete ML lifecycle—from data preprocessing and model training to deployment and user interaction.

The core model is a Random Forest regressor trained on preprocessed candidate and industry data. It achieves **99.82% variance explained (R² = 0.9982)** with minimal prediction error (**RMSE = 0.0267**), making it reliable for salary estimation across different roles and experience levels.

The web interface allows recruiters, HR professionals, and candidates to input relevant attributes and receive immediate salary predictions in Indian Rupees (LPA).

---

## 🚀 Live Demo

Access the deployed application here:  
**https://salary-prediction-capstone.onrender.com**

The application is hosted on Render with automatic GitHub integration for continuous deployment.

---

## ✨ Features

- **Interactive Web Interface** — Clean, responsive form-based UI built with HTML and CSS
- **Real-time Predictions** — Instant salary estimates based on candidate input
- **Multi-factor Analysis** — Considers experience, education, skills, role, and industry
- **Domain-Specific Adjustments** — Applies industry and role-based modifiers to base predictions
- **Persistent Model** — Trained model saved as pickle for production use
- **Scalable Deployment** — Runs on Gunicorn with Flask backend on Render

---

## 📸 Project Showcase

<details open>
<summary><b>Expand to view visualizations</b></summary>

### Model Performance

**Feature Importance Analysis**

The most influential predictors are visualized to understand model behavior and validate business logic.

```
Screenshot: feature_importance.png
(Located in /screenshots)
```

**SHAP Explainability**

SHAP values provide individual prediction explanations, ensuring model transparency.

```
Screenshot: shap_summary.png
(Located in /screenshots)
```

</details>

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.8+ | Core development |
| **ML Framework** | Scikit-learn | Model training & inference |
| **Algorithm** | Random Forest Regressor | Regression task |
| **Data Processing** | Pandas, NumPy | Preprocessing & transformation |
| **Backend** | Flask | REST API & server |
| **Frontend** | HTML, CSS | User interface |
| **Model Serialization** | Joblib, Pickle | Model persistence |
| **Deployment** | Render + Gunicorn | Production hosting |
| **Version Control** | Git, GitHub | Code management |

---

## 🤖 Machine Learning Pipeline

The project follows a structured ML workflow:

```
1. Data Preparation
   ├── Load raw salary data
   ├── Handle missing values
   └── Exploratory data analysis

2. Feature Engineering
   ├── Categorical encoding (one-hot)
   ├── Numerical scaling
   └── Feature selection

3. Model Training
   ├── Train-test split
   ├── Random Forest model training
   ├── Hyperparameter tuning
   └── Cross-validation

4. Model Evaluation
   ├── RMSE calculation
   ├── R² score
   ├── Feature importance analysis
   └── SHAP explainability

5. Model Deployment
   ├── Serialize to pickle
   ├── Create Flask API
   ├── Build web interface
   └── Deploy to Render
```

---

## 📊 Model Information

### Algorithm
**Random Forest Regressor** — An ensemble learning method combining multiple decision trees for robust predictions.

### Hyperparameters
```python
n_estimators = 50        # Number of trees
max_depth = 15           # Maximum tree depth
random_state = 42        # Reproducibility
n_jobs = -1              # Parallel processing
```

### Training Data
- **Source**: Preprocessed tabular salary dataset
- **Features**: One-hot encoded categorical variables
- **Target**: Cost To Company (CTC) in LPA

### Model Performance

| Metric | Score |
|--------|-------|
| **R² Score** | 0.9982 |
| **RMSE** | 0.0267 |
| **Model File** | `models/random_forest.pkl` |

The model explains **99.82%** of variance in salary predictions with minimal error margin.

---

## 📁 Project Structure

```
Salary-Prediction_capstone/
│
├── app/
│   └── app.py                    # Flask application & routes
│
├── templates/
│   └── index.html                # HTML form interface
│
├── models/
│   ├── random_forest.pkl         # Trained model (pickle)
│   ├── random_forest_model.joblib # Model backup (joblib)
│   └── freq_maps.joblib          # Frequency encodings
│
├── data/
│   └── expected_ctc.csv          # Original salary dataset
│
├── notebook/
│   ├── 01_data_cleaning.py       # Data preprocessing
│   ├── 02_model_training.py      # Model development
│   ├── 03_model_explainability.py # SHAP analysis
│   ├── X_train.csv               # Training features
│   ├── X_test.csv                # Test features
│   ├── y_train.csv               # Training targets
│   └── y_test.csv                # Test targets
│
├── screenshots/
│   ├── feature_importance.png    # Model insights
│   └── shap_summary.png          # Explainability plots
│
├── requirements.txt              # Python dependencies
├── Procfile                       # Render deployment config
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

---

## 🔧 Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bhuvan-TNL/Salary-Prediction_capstone.git
   cd Salary-Prediction_capstone
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**
   Ensure `models/random_forest.pkl` exists in the project directory. This is the trained model required for predictions.

---

## 🏃 Running Locally

### Start the Flask Application

```bash
cd app
python app.py
```

The application will start on `http://localhost:5000`

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

You should see the salary prediction form. Enter candidate attributes and click "Predict" to see the estimated CTC.

---

## 🌐 Deployment Process

### Deployment Architecture

```
┌─────────────┐
│   GitHub    │  Push code
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│     Render      │  Auto-deploy on push
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│    Gunicorn      │  WSGI application server
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      Flask       │  Web framework & routes
└──────────────────┘
```

### Deployment Steps

1. **Push to GitHub**
   - Commit changes locally
   - Push to `main` branch

2. **Render Configuration**
   - Connect GitHub repository to Render
   - Render reads `Procfile` and `requirements.txt`
   - Automatically installs dependencies

3. **Server Start**
   - Render executes Procfile command
   - Gunicorn launches Flask app
   - Application is live

4. **Continuous Deployment**
   - Each push to `main` triggers automatic redeployment
   - Downtime is minimal with rolling deployments

---

## 🔌 API Routes

### GET /
**Home endpoint** — Serves the prediction form interface.

```bash
curl http://localhost:5000/
```

**Response**: HTML page with input form

---

### POST /predict
**Prediction endpoint** — Accepts candidate data and returns salary estimate.

**Request Body** (Form Data):
```
total_experience   : integer (years)
field_experience   : integer (years)
degree             : string (e.g., "B.Tech", "M.Tech")
skills             : comma-separated string
job_role           : string (e.g., "Software Engineer", "Data Scientist")
industry           : string (e.g., "IT", "Finance")
```

**cURL Example**:
```bash
curl -X POST http://localhost:5000/predict \
  -d "total_experience=5&field_experience=3&degree=B.Tech&skills=Python,ML,SQL&job_role=ML Engineer&industry=IT"
```

**Response**:
```json
{
  "predicted_salary": 15.5,
  "unit": "LPA"
}
```

---

## 📝 Example Prediction Workflow

### Scenario
A candidate has:
- **Total Experience**: 5 years
- **Field Experience**: 3 years
- **Degree**: B.Tech
- **Skills**: Python, Machine Learning, SQL
- **Role**: ML Engineer
- **Industry**: IT

### Processing Steps

1. **Input Validation**
   - Form captures all required fields

2. **Encoding**
   - Categorical variables (degree, role, industry) are one-hot encoded
   - Matches training data format

3. **Model Inference**
   - Pre-trained Random Forest model loads from `models/random_forest.pkl`
   - Prediction generated on encoded features

4. **Domain Adjustments**
   - Base prediction adjusted for:
     - Total years of experience (multiplier increases with seniority)
     - Field-specific experience (domain expertise premium)
     - Number of relevant skills (technical depth bonus)
     - Role-specific market rates
     - Industry salary benchmarks

5. **Output**
   - Final CTC displayed in LPA (Lakhs Per Annum)
   - Result shown on web interface

### Example Output
```
Predicted CTC: 15.5 LPA
```

---

## 🔮 Future Improvements

- **API Authentication** — Secure endpoints with API keys for programmatic access
- **Batch Predictions** — Accept multiple candidates in a single request
- **Advanced Visualizations** — Interactive charts showing salary distribution by role/industry
- **Model Versioning** — Track model performance over time with A/B testing
- **Data Monitoring** — Log predictions for future model retraining
- **Mobile Optimization** — Responsive design for mobile devices
- **Additional Features** — Incorporate certifications, years to promotion, skill specialization
- **Explainability UI** — Show SHAP values for individual predictions
- **Multi-region Support** — Salary predictions for different countries/currencies
- **Real-time Model Retraining** — Automated pipelines with fresh data

---

## 📚 Learning Outcomes

This project demonstrates practical expertise in:

✅ **Machine Learning**
- Regression modeling with ensemble methods
- Hyperparameter tuning and cross-validation
- Model evaluation metrics (R², RMSE)
- Feature importance analysis

✅ **Data Engineering**
- Data cleaning and preprocessing
- Categorical encoding
- Train-test splitting and stratification
- Data pipeline design

✅ **Software Engineering**
- Flask application development
- API design and RESTful principles
- Model serialization and loading
- Front-end integration

✅ **Deployment & DevOps**
- Containerized application setup (Gunicorn)
- Git-based continuous integration
- Cloud deployment on Render
- Production environment configuration

✅ **Model Explainability**
- SHAP value analysis
- Feature importance visualization
- Business logic validation

---

## 👤 Author

**Bhuvanesh**

Computer Science Engineering student at Chandigarh University  
Specialization: Artificial Intelligence and Technology  
AI/ML Internship: LaunchedGlobal.in

- **GitHub**: [Bhuvan-TNL](https://github.com/Bhuvan-TNL)
- **LinkedIn**: [Connect](https://www.linkedin.com/in/bhuvanesh-bezawada/)

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

You are free to use, modify, and distribute this project for personal or commercial purposes.

---

## 🙏 Acknowledgements

- **Scikit-learn** — For the powerful Random Forest algorithm and preprocessing utilities
- **Flask** — For the lightweight web framework
- **Pandas & NumPy** — For data manipulation and numerical computing
- **Render** — For seamless cloud deployment
- **SHAP** — For model explainability insights

---

<div align="center">

**Made with ❤️ | Open to Feedback and Contributions**

If you found this project useful, please consider giving it a ⭐ on GitHub!

</div>
