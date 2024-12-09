import pandas as pd

def new_features(data: pd.DataFrame):
    app_efficiency = data["App Usage Time (min/day)"] / data["Number of Apps Installed"]

    battery_efficiency = data["Screen On Time (hours/day)"] / data["Battery Drain (mAh/day)"]

    usage_intensity = (data["Screen On Time (hours/day)"] * data["App Usage Time (min/day)"]) / (24 * 60)

    app_usage_density = data["App Usage Time (min/day)"] / data["Screen On Time (hours/day)"]

    return {"App Efficiency": app_efficiency.round(3),
            "Battery Efficiency": battery_efficiency.round(3),
            "Usage Intensity": usage_intensity.round(3),
            "App Usage Density": app_usage_density.round(3)}