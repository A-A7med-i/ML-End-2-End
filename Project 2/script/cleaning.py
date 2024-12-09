import pandas as pd

data_engineering = [
    "Data Engineer",
    "Data Analyst",
    "Analytics Engineer",
    "BI Data Analyst",
    "Business Data Analyst",
    "BI Developer",
    "BI Analyst",
    "Business Intelligence Engineer",
    "BI Data Engineer",
    "Power BI Developer",
]


data_scientist = [
    "Data Scientist",
    "Applied Scientist",
    "Research Scientist",
    "3D Computer Vision Researcher",
    "Deep Learning Researcher",
    "AI/Computer Vision Engineer",
]


machine_learning = [
    "Machine Learning Engineer",
    "ML Engineer",
    "Lead Machine Learning Engineer",
    "Principal Machine Learning Engineer",
]


data_architecture = [
    "Data Architect",
    "Big Data Architect",
    "Cloud Data Architect",
    "Principal Data Architect",
]


management = [
    "Data Science Manager",
    "Director of Data Science",
    "Head of Data Science",
    "Data Scientist Lead",
    "Head of Machine Learning",
    "Manager Data Management",
    "Data Analytics Manager",
]


def classify_job_title(row) -> str:
    if row in data_engineering:
        return "Data Engineering"
    elif row in data_scientist:
        return "Data Science"
    elif row in machine_learning:
        return "Machine Learning"
    elif row in data_architecture:
        return "Data Architecture"
    elif row in management:
        return "Management"
    else:
        return "Other"



def classify_job_titles(data: pd.DataFrame) -> pd.Series:
    return data["job_title"].apply(lambda x: classify_job_title(x))



def refactor_experience_level(data:pd.DataFrame) -> pd.DataFrame:
    return data["experience_level"].replace(
        {
            'SE': 'Senior',
            'EN': 'Entry level',
            'EX': 'Executive level',
            'MI': 'Mid/Intermediate level',
        }
    )

def refactor_employment_type(data:pd.DataFrame) -> pd.DataFrame:
    return data["employment_type"].replace(
        {
            'FL': 'Freelancer',
            'CT': 'Contractor',
            'FT': 'Full-time',
            'PT': 'Part-time'
        }
    )

def refactor_company_size(data:pd.DataFrame)->pd.DataFrame:
    return data["company_size"].replace(
        {
            'S': 'SMALL',
            'M': 'MEDIUM',
            'L': 'LARGE',
        }
    )

def refactor_remote_ratio(data:pd.DataFrame)->pd.DataFrame:
    return data["remote_ratio"].replace(
        {
            0: 'On-Site',
            50: 'Half-Remote',
            100: 'Full-Remote',
        }
    )
