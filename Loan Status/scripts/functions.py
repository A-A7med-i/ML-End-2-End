import pandas as pd

def filling_dependents(row: pd.Series, mask_a: list[float], mask_c: list[float]) -> int:
    """
    Determines the number of dependents based on income thresholds.

    Args:
        row (pd.Series): A row from a DataFrame containing income data.
        mask_a (list[float]): A list of income thresholds for the applicant.
        mask_c (list[float]): A list of income thresholds for the coapplicant.

    Returns:
        int: The estimated number of dependents.
    """
    if row["CoapplicantIncome"] < mask_c[0] and row["ApplicantIncome"] < mask_a[0]:
        return 0

    elif row["CoapplicantIncome"] < mask_c[1] and row["ApplicantIncome"] < mask_a[1]:
        return 1

    elif row["CoapplicantIncome"] < mask_c[2] and row["ApplicantIncome"] < mask_a[2]:
        return 2

    else:
        return 3


def filling_self_employed(row: pd.Series, val_1: float, val_2: float) -> str:
    """
    Determines whether the applicant is self-employed based on income thresholds.

    Args:
        row (pd.Series): A row from a DataFrame containing income data.
        val_1 (float): An income threshold for the applicant.
        val_2 (float): An income threshold for the coapplicant.

    Returns:
        str: "Yes" if the applicant is self-employed, "No" otherwise.
    """
    if row["ApplicantIncome"] > val_1 and row["CoapplicantIncome"] < val_2:
        return "Yes"

    else:
        return "No"


def filling_loan_term(row: pd.Series, data: pd.DataFrame) -> float:
    """
    Determines the loan term based on the loan amount.

    Args:
        row (pd.Series): A row from a DataFrame containing loan amount data.
        data (pd.DataFrame): The entire DataFrame containing loan amount data.

    Returns:
        float: The loan term in months.
    """
    if row["LoanAmount"] > data["LoanAmount"].mean():
        return 360.0

    else:
        return 180.0


def filling_credit_history(row: pd.Series) -> float:
    """
    Determines the credit history based on the loan status.

    Args:
        row (pd.Series): A row from a DataFrame containing loan status data.

    Returns:
        float: 1.0 if the loan status is "Y", 0.0 otherwise.
    """

    if row["Loan_Status"] == "Y":
        return 1.0
    else:
        return 0.0
