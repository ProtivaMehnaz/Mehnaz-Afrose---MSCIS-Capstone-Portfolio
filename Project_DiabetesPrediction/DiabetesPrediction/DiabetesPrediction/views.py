from django.shortcuts import render
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def bmi(request):
    return render(request, 'bmi.html')

def predict(request):
    return render(request, 'predict.html')

def suggestion(request):
    return render(request, 'suggestion.html')

def result(request):
    data = pd.read_csv(r"new_dataset.csv")

    # Drop the 'diabetes', 'gender', and 'smoking_history' columns from the features (X)
    X = data.drop(["diabetes"], axis=1)
    Y = data['diabetes']

    if os.path.exists('rf_model.pkl'):
        # Load the trained model
        rf_classifier = joblib.load('rf_model.pkl')
    else:
        # Train the model if it is not trained
        # Split the data
        X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)
        # Initialize the Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)
        # Train the model on the training data
        rf_classifier.fit(X_train, Y_train)
        # Save the trained model
        joblib.dump(rf_classifier, 'rf_model.pkl')


    val1 = request.GET['n1']
    val2 = request.GET['n2']
    val3 = request.GET['n3']
    val4 = request.GET['n4']
    val5 = request.GET['n5']
    height = float(request.GET['height'])
    weight = float(request.GET['weight'])

    val6 = weight/(height*height)
    val7 = request.GET['n7']
    val8 = request.GET['n8']

    input_data = [[val1, val2, val3, val4, val5, val6, val7, val8]]
    pred = rf_classifier.predict(input_data)

    result1 = ""
    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"


    return render (request, 'predict.html', {"result2":result1})

