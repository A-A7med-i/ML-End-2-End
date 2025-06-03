# Healthcare Disease Prediction

An end-to-end machine learning project for predicting diseases based on symptoms, featuring a custom web interface for user interaction.

---

## Project Overview

This project covers the full ML workflow:

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model development and evaluation
- Model deployment with a Flask-based web app

---

## Directory Structure

```text
├── data/
│   ├── raw/         # Raw data files
│   └── processed/   # Cleaned and processed data
├── deployment/
│   └── app.py       # Flask app for model deployment
│   └── static/      # Static files (CSS, JS)
│   └── templates/   # HTML templates
├── model/           # Trained model artifacts (e.g., model.pkl, encoder.pkl)
├── modeling/        # Model training scripts
├── notebooks/       # Jupyter notebooks for EDA and modeling
├── reports/         # Project reports and visualizations
├── scripts/         # Data processing and utility scripts
├── requirements.txt # Python dependencies
├── setup.py         # Project setup
└── README.md        # Project documentation
```

---

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/A-A7med-i/ML-End-2-End
   cd 'Health Care'
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python deployment/app.py
   ```

   The app will be available at the local URL provided in the terminal.

---

## Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript (via Flask templates)
- **Machine Learning:** scikit-learn

---

## Notebooks

- Step-by-step EDA, feature engineering, and model development can be found in the `notebooks/` directory.

---

## Contributing

We welcome contributions!

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit and push your branch
5. Open a pull request

---
