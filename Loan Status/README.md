# Loan Status Prediction Web App

An end-to-end machine learning application that predicts loan approval status based on applicant features. The model is deployed using Streamlit for an interactive web interface.

---

## Project Overview

This project demonstrates the complete ML workflow:

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model development and evaluation
- Model deployment with Streamlit (interactive web app)

---

## Features

- Interactive web interface for loan prediction
- Real-time prediction using a trained ML model
- Input validation and data preprocessing
- Responsive design for desktop and mobile devices

---

## Directory Structure

```text
├── data/
│   ├── raw/         # Raw data files
│   └── processed/   # Cleaned and processed data
├── deployment/
│   └── app.py       # Streamlit app for model deployment
├── model/           # Trained model artifacts (e.g., model.pkl)
├── modeling/        # Model training scripts
├── notebook/        # Jupyter notebooks for EDA and modeling
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
   cd 'Loan Status'
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
   streamlit run deployment/app.py
   ```

   The app will be available at the local URL provided in the terminal.

---

## Technologies Used

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- NumPy
- joblib

---

## Notebooks

- Step-by-step EDA, feature engineering, and model development can be found in the `notebook/` directory.

---

## Contributing

We welcome contributions!

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit and push your branch
5. Open a pull request
