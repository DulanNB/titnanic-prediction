# Titanic Survival Prediction Web Application

This repository contains code for a web application that predicts the survival likelihood of passengers on the Titanic using machine learning algorithms. The application is built in Python and utilizes Flask framework for the backend.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python (version 3.x)
- Flask (you can install it via pip: `pip install Flask`)

## Running the Application

1. Clone this repository to your local machine:

```
git clone https://github.com/your-username/titnanic-prediction.git
```

2. Navigate to the directory containing the repository:

```
cd titanic-survival-prediction
```

3. Run the Flask application:

```
python app.py
```

This will start the Flask server. You should see output indicating that the server is running and listening for requests.

4. Open a web browser and go to the following URL:

```
http://127.0.0.1:5000/ or Your Local url
```

This will take you to the web interface of the application.

## Making Predictions

- On the web interface, you'll see input fields where you can enter information about a passenger (such as age, gender, class, etc.).
- Fill in the required information and click the "Predict" button.
- The application will use a pre-trained machine learning model to predict the likelihood of survival for the passenger based on the input provided.
- You will receive the prediction result on the web page.

## Files Description

- `titanic.py`: This file contains the code for loading the pre-trained machine learning model and making predictions.
- `app.py`: This is the main Flask application file. It contains the routes and logic for handling requests and rendering web pages.
- `templates/`: This directory contains HTML templates used by Flask to render web pages.
- `static/`: This directory contains static files (like CSS and JavaScript) used by the web application.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
