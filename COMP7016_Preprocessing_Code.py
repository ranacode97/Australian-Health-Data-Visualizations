
import pandas as pd
import numpy as np

# Load Excel file
xls = pd.ExcelFile("Data_Visualisation_Assignment2.xlsx")
s4_3 = xls.parse('Table S4.3', skiprows=2)
s4_4 = xls.parse('Table S4.4', skiprows=2)

# Rename columns for easier access
s4_3.columns = [...]
s4_4.columns = [...]

# Impute missing counts from percentages
def impute_count(count, percent, total):
    mask = count.astype(str).str.contains("n.p.")
    count[mask] = (pd.to_numeric(percent[mask], errors='coerce') / 100) * pd.to_numeric(total[mask], errors='coerce')
    return pd.to_numeric(count, errors='coerce').round(0)

# Impute missing percentages from counts
def impute_percent(percent, count, total):
    mask = percent.astype(str).str.contains("n.p.")
    percent[mask] = (pd.to_numeric(count[mask], errors='coerce') / pd.to_numeric(total[mask], errors='coerce')) * 100
    return pd.to_numeric(percent, errors='coerce').round(2)

# Fill missing values in S4.3
s4_3["Ambulance_Arrivals"] = impute_count(s4_3["Ambulance_Arrivals"], s4_3["Ambulance_%"], s4_3["Total_ED_Presentations"])
s4_3["Police_Arrivals"] = impute_count(s4_3["Police_Arrivals"], s4_3["Police_%"], s4_3["Total_ED_Presentations"])
s4_3["Other_Arrivals"] = impute_count(s4_3["Other_Arrivals"], s4_3["Other_%"], s4_3["Total_ED_Presentations"])
s4_3["Police_%"] = impute_percent(s4_3["Police_%"], s4_3["Police_Arrivals"], s4_3["Total_ED_Presentations"])
s4_3["Other_%"] = impute_percent(s4_3["Other_%"], s4_3["Other_Arrivals"], s4_3["Total_ED_Presentations"])
s4_3["Unknown_Arrivals"] = s4_3["Total_ED_Presentations"] - (
    s4_3["Ambulance_Arrivals"] + s4_3["Police_Arrivals"] + s4_3["Other_Arrivals"]
)

# For S4.4: Impute subconditions from average shares, scaled to match total
condition_cols = ["Pneumonia", "Diabetes", "Anaemia", "UTI", "Dental_Cond"]
valid_rows = s4_4[condition_cols + ["Preventable_Hosp"]].dropna()
valid_rows = valid_rows[valid_rows[condition_cols].sum(axis=1) <= valid_rows["Preventable_Hosp"]]
avg_shares = valid_rows[condition_cols].div(valid_rows["Preventable_Hosp"], axis=0).mean()

# Proportional imputation with scaling to fit total
for idx, row in s4_4.iterrows():
    known = row[condition_cols].dropna()
    unknown = row[condition_cols].isna()
    total = row["Preventable_Hosp"]
    if unknown.any():
        remaining = total - known.sum()
        shares = avg_shares[unknown.index]
        norm_shares = shares / shares.sum()
        for cond in unknown.index:
            s4_4.at[idx, cond] = round(norm_shares[cond] * remaining)

# Final scaling if sum still exceeds total
for idx, row in s4_4.iterrows():
    subtotal = row[condition_cols].sum()
    total = row["Preventable_Hosp"]
    if subtotal > total:
        scale = total / subtotal
        for cond in condition_cols:
            s4_4.at[idx, cond] = round(row[cond] * scale)
