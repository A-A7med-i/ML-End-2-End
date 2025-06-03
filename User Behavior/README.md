# User Behavior Analytics End-to-End Project

A web application that tracks and analyzes user behavior using a FastAPI backend and a simple HTML/CSS/JS frontend.

---

## Project Overview

This project demonstrates the complete ML workflow:

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model development and evaluation
- Model deployment with FastAPI (REST API)
- Frontend for user interaction and visualization

---

## Directory Structure

```text
├── data/
│   ├── raw/         # Raw user behavior data
│   └── processed/   # Cleaned and processed data
├── deployment/
│   └── main.py      # FastAPI app for serving the model
│   └── static/      # Static files (CSS, JS)
│   └── templates/   # HTML templates
├── modeling/        # Model training scripts
├── notebook/        # Jupyter notebooks for EDA and modeling
├── reports/         # Project reports and visualizations
├── saved-model/     # Trained model artifacts
├── script/          # Data processing and utility scripts
├── requirements.txt # Python dependencies
├── setup.py         # Project setup
└── README.md        # Project documentation
```

---

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/A-A7med-i/ML-End-2-End
   cd 'User Behavior'
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
   uvicorn deployment.main:app --reload
   ```

   The app will be available at the local URL provided in the terminal.

---

## Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** scikit-learn

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
