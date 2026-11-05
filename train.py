
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("college_student_placement_dataset.csv")

# Remove College_ID
df = df.drop(columns=["College_ID"])

# Convert categorical values into numerical values
df["Internship_Experience"] = df["Internship_Experience"].map(
    {"Yes": 1, "No": 0}
)

df["Placement"] = df["Placement"].map(
    {"Yes": 1, "No": 0}
)

# Separate features and target
X = df.drop("Placement", axis=1)
y = df["Placement"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

model.fit(X_train, y_train)

# Predict placement
predictions = model.predict(X_test)

# Evaluate model
print("Model Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Student Placement Model Training Successful!")
