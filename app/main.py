from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("movies_success_model.pkl")  # Ensure this file exists in the same directory or provide a full path

# Define the input data schema
class MovieData(BaseModel):
    budget: float
    runtime: float
    profit_margin: float
    release_month: int
    Action: int
    Adventure: int
    Animation: int
    Comedy: int
    Crime: int
    Documentary: int
    Drama: int
    Family: int
    Fantasy: int
    Foreign: int
    History: int
    Horror: int
    Music: int
    Mystery: int
    Romance: int
    Science_Fiction: int
    TV_Movie: int
    Thriller: int
    War: int
    Western: int

# Endpoint for prediction
@app.post("/predict/")
def predict(movie_data: MovieData):
    # Convert the input Pydantic model to a dictionary
    input_data = movie_data.dict()

    # Replace keys to match the model feature names
    input_data["Science Fiction"] = input_data.pop("Science_Fiction")
    input_data["TV Movie"] = input_data.pop("TV_Movie")

    # Ensure all values are native Python types
    formatted_data = {key: int(value) if isinstance(value, (np.integer, bool))
                      else float(value) if isinstance(value, np.floating)
                      else value
                      for key, value in input_data.items()}

    # Convert dictionary values into a 2D array
    input_array = [list(formatted_data.values())]

    # Predict using the model
    prediction = model.predict(input_array)

    # Return prediction in JSON format
    return jsonable_encoder({"prediction": int(prediction[0])})
