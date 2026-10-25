# utils.py
import joblib
import pandas as pd
import numpy as np

def load_trained_model():
    try:
        # Load your trained model from the root folder
        model = joblib.load('car_price_model.pkl')
        return model
    except Exception as e:
        return None

def predict_price(model, input_data):
    if model is not None:
        # If your actual model is loaded, use this:
        # prediction = model.predict(input_data)[0]
        # return prediction
        pass
    
    # Fallback placeholder calculation if model isn't connected yet
    return 500000