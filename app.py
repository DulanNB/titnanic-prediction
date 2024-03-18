from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load the trained model and LabelEncoders
clf = joblib.load("titanic_model.joblib")
le_sex = joblib.load("le_sex.joblib")
le_embarked = joblib.load("le_embarked.joblib")

# Initialize Flask app
app = Flask(__name__)

# Define preprocessing function
def preprocess_input(data):
    # Fill missing values
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna("U")
    
    # Encode categorical variables
    data["Sex"] = le_sex.transform(data["Sex"])
    data["Embarked"] = le_embarked.transform(data["Embarked"])
    
    return data

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        passenger_class = int(request.form['Pclass'])
        sex = request.form['Sex']
        age = float(request.form['Age'])
        siblings_spouses = int(request.form['SibSp'])
        parents_children = int(request.form['Parch'])
        fare = float(request.form['Fare'])
        embarked = request.form['Embarked']

        # Create DataFrame with form data
        input_data = pd.DataFrame({
            "Pclass": [passenger_class],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [siblings_spouses],
            "Parch": [parents_children],
            "Fare": [fare],
            "Embarked": [embarked]
        })

      
        # Preprocess input data
        input_data = preprocess_input(input_data)

        print("Input Data:")
        print(input_data)


        # Make prediction
        prediction = clf.predict(input_data)

        # Display prediction result
        if prediction[0] == 1:
            result = "Survived"
        else:
            result = "Did not survive"

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
