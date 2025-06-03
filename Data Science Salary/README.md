# Data Science Salary Predictor

An end-to-end machine learning project that predicts data science salaries based on features such as experience, location, skills, and company size.

---

## Project Overview

This project demonstrates the full ML workflow:

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model development and evaluation
- Model deployment with Gradio (interactive web app)

---

## Directory Structure

```text
├── data/
│   ├── raw/         # Raw data files
│   └── processed/   # Cleaned and processed data
├── deployment/
│   └── app.py       # Gradio app for model deployment
├── model/           # Trained model artifacts (e.g., random_forest.pkl)
├── modeling/        # Model training scripts
├── notebook/        # Jupyter notebooks for EDA and modeling
├── reports/         # Project reports and visualizations
├── script/          # Data processing and utility scripts
├── requirements.txt # Python dependencies
├── setup.py         # Project setup
└── README.md        # Project documentation
```

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/A-A7med-i/ML-End-2-End
   cd 'Data Science Salary'
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

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- gradio
- joblib

---

## Usage & Deployment

The model is deployed using Gradio, providing an interactive web interface.

To run the application locally:

```bash
python deployment/app.py
```

Then open the provided local URL in your browser to interact with the salary predictor.

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
