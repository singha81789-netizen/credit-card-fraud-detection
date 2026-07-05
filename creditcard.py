
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")




df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


duplicate_count = df.duplicated().sum()
print("\nDuplicate Rows:", duplicate_count)


df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)

actual_labels = df["Class"]


df["Transaction_ID"] = df.index


df["Amount_Log"] = np.log1p(df["Amount"])


df["Hour"] = (df["Time"] / 3600) % 24


df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

print("\nNew Features Created:")
print(df[["Time", "Hour", "Hour_Sin", "Hour_Cos", "Amount", "Amount_Log"]].head())


plt.figure(figsize=(8, 5))
sns.histplot(df["Amount"], bins=100, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df["Amount_Log"], bins=100, kde=True)
plt.title("Log Transformed Transaction Amount Distribution")
plt.xlabel("Log Amount")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Amount"])
plt.title("Transaction Amount Boxplot")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df["Hour"], bins=24, kde=True)
plt.title("Transaction Time Distribution")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x=df["Class"])
plt.title("Class Distribution (Only for Dataset Understanding)")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

print("\nClass Distribution:")
print(df["Class"].value_counts())




feature_columns = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount_Log", "Hour_Sin", "Hour_Cos"]
)

X = df[feature_columns]

print("\nSelected Features Shape:", X.shape)
print("\nSelected Features:")
print(feature_columns)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed.")
print("Scaled Data Shape:", X_scaled.shape)



isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(X_scaled)

print("\nIsolation Forest Model Training Completed.")




df["Anomaly_Label"] = isolation_forest.predict(X_scaled)

df["Anomaly_Score"] = isolation_forest.decision_function(X_scaled)

df["Status"] = df["Anomaly_Label"].map({
    1: "Normal",
    -1: "Suspicious"
})

print("\nAnomaly Detection Results:")
print(df["Status"].value_counts())



high_amount_threshold = df["Amount"].quantile(0.99)


hour_counts = df["Hour"].round().value_counts()
rare_hours = hour_counts[hour_counts < hour_counts.quantile(0.25)].index.tolist()

def get_reason(row):
    reasons = []

    if row["Amount"] > high_amount_threshold:
        reasons.append("Unusually high transaction amount")

    if round(row["Hour"]) in rare_hours:
        reasons.append("Unusual transaction time")

    if row["Anomaly_Score"] < df["Anomaly_Score"].quantile(0.01):
        reasons.append("Very unusual transaction pattern")

    if len(reasons) == 0 and row["Status"] == "Suspicious":
        reasons.append("Different from normal transaction behavior")

    if len(reasons) == 0:
        return "Normal transaction pattern"

    return ", ".join(reasons)

df["Reason"] = df.apply(get_reason, axis=1)




suspicious_transactions = df[df["Status"] == "Suspicious"].copy()


top_suspicious = suspicious_transactions.sort_values(
    by="Anomaly_Score",
    ascending=True
)

final_columns = [
    "Transaction_ID",
    "Time",
    "Hour",
    "Amount",
    "Amount_Log",
    "Anomaly_Score",
    "Anomaly_Label",
    "Status",
    "Reason"
]

print("\nTop 20 Most Suspicious Transactions:")
print(top_suspicious[final_columns].head(20))




top_suspicious[final_columns].to_csv(
    "suspicious_transactions.csv",
    index=False
)

print("\nFile saved successfully: suspicious_transactions.csv")





pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PCA_1"] = X_pca[:, 0]
df["PCA_2"] = X_pca[:, 1]

plt.figure(figsize=(10, 7))

normal_data = df[df["Status"] == "Normal"]
anomaly_data = df[df["Status"] == "Suspicious"]

plt.scatter(
    normal_data["PCA_1"],
    normal_data["PCA_2"],
    alpha=0.3,
    label="Normal Transactions"
)

plt.scatter(
    anomaly_data["PCA_1"],
    anomaly_data["PCA_2"],
    alpha=0.8,
    label="Suspicious Transactions"
)

plt.title("PCA Visualization of Normal and Suspicious Transactions")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()



validation_table = pd.crosstab(
    df["Class"],
    df["Status"],
    rownames=["Actual Class"],
    colnames=["Model Prediction"]
)

print("\nOptional Validation Table:")
print(validation_table)

fraud_detected = df[
    (df["Class"] == 1) &
    (df["Status"] == "Suspicious")
].shape[0]

total_fraud = df[df["Class"] == 1].shape[0]

fraud_detection_rate = (fraud_detected / total_fraud) * 100

print("\nKnown Fraud Transactions:", total_fraud)
print("Known Fraud Detected as Suspicious:", fraud_detected)
print("Fraud Detection Rate:", round(fraud_detection_rate, 2), "%")


total_transactions = len(df)
total_anomalies = len(suspicious_transactions)
anomaly_percentage = (total_anomalies / total_transactions) * 100

print("\n========== FINAL PROJECT SUMMARY ==========")
print("Total Transactions:", total_transactions)
print("Suspicious Transactions Detected:", total_anomalies)
print("Anomaly Percentage:", round(anomaly_percentage, 2), "%")
print("Model Used: Isolation Forest")
print("Training Type: Unsupervised Learning")
print("===========================================")