import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

start_time = time.time()

# Load and preprocess the data
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["PassengerId"]

# Function to clean the data
def clean(data):
    data = data.drop(["Ticket", "PassengerId", "Name", "Cabin"], axis=1)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna("U")
    return data

# Clean the data
data = clean(data)
test = clean(test)

# Encode categorical variables
le_sex = LabelEncoder()
data["Sex"] = le_sex.fit_transform(data["Sex"])
test["Sex"] = le_sex.transform(test["Sex"])

le_embarked = LabelEncoder()
data["Embarked"] = le_embarked.fit_transform(data["Embarked"])
test["Embarked"] = le_embarked.transform(test["Embarked"])

# Split data into features and target variable
y = data["Survived"]
X = data.drop("Survived", axis=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(random_state=0)

start_train_time = time.time()
clf.fit(X_train, y_train)
end_train_time = time.time()

# Predict on validation data
val_predictions = clf.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)

# Predict on test data
start_pred_time = time.time()
test_predictions = clf.predict(test)
end_pred_time = time.time()

# Save the trained model and LabelEncoders
joblib.dump(clf, "titanic_model.joblib")
joblib.dump(le_sex, "le_sex.joblib")
joblib.dump(le_embarked, "le_embarked.joblib")

# Save test predictions
df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": test_predictions,
                  })

df.to_csv("submission.csv", index=False)

end_time = time.time()
print("Training Time:", end_train_time - start_train_time)
print("Prediction Time:", end_pred_time - start_pred_time)
print("Total Execution Time:", end_time - start_time)
