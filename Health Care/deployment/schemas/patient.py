from pydantic import BaseModel, Field
from typing import Dict, Union
from decimal import Decimal
from datetime import date


class PatientData(BaseModel):
    """
    A Pydantic model representing patient data.

    Attributes:
        Age: Patient's age (integer, greater than 0 and less than 150).
        Gender: Patient's gender (string).
        Blood_Type: Patient's blood type (string).
        Medical_Condition: Patient's medical condition (string).
        Date_of_Admission: Date of admission (date object).
        Insurance_Provider: Patient's insurance provider (string).
        Billing_Amount: Billing amount (decimal, greater than or equal to 0).
        Room_Number: Room number (integer, greater than 0).
        Admission_Type: Admission type (string).
        Discharge_Date: Date of discharge (date object).
        Medication: Medications prescribed (string).
    """

    Age: int = Field(..., gt=0, lt=150)
    Gender: str
    Blood_Type: str = Field(..., alias="Blood Type")
    Medical_Condition: str = Field(..., alias="Medical Condition")
    Date_of_Admission: date = Field(..., alias="Date of Admission")
    Insurance_Provider: str = Field(..., alias="Insurance Provider")
    Billing_Amount: Decimal = Field(..., ge=0, alias="Billing Amount")
    Room_Number: int = Field(..., gt=0, alias="Room Number")
    Admission_Type: str = Field(..., alias="Admission Type")
    Discharge_Date: date = Field(..., alias="Discharge Date")
    Medication: str

    def to_dict(self) -> Dict[str, Union[str, int, float, date]]:
        """
        Converts the PatientData object to a dictionary.

        Returns:
            A dictionary representing the patient data.
        """
        return {
            "Age": self.Age,
            "Gender": self.Gender,
            "Blood Type": self.Blood_Type,
            "Medical Condition": self.Medical_Condition,
            "Date of Admission": str(self.Date_of_Admission),
            "Insurance Provider": self.Insurance_Provider,
            "Billing Amount": float(self.Billing_Amount),
            "Room Number": self.Room_Number,
            "Admission Type": self.Admission_Type,
            "Discharge Date": str(self.Discharge_Date),
            "Medication": self.Medication,
        }
