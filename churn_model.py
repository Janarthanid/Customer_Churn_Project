import pandas as pd

# Load dataset
df = pd.read_csv("data/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Show top 5 rows
print(df.head())

# Check info
print(df.info())

# Check missing values
print(df.isnull().sum())

# Drop the ID column (not needed for training)
df.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric (some have blanks)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert 'Churn' column to numbers (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)  # features
y = df['Churn']               # target column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns

feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

import joblib
joblib.dump(model, "churn_model.pkl")
print("✅ Model saved as churn_model.pkl")

