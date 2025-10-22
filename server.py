from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI()

# Templates directory
templates = Jinja2Templates(directory="templates")

# Load model and feature columns
model = joblib.load("house_price_prediction_model.joblib")
columns = np.load("columns.npy", allow_pickle=True)
areatype = np.load("areatypes.npy", allow_pickle=True)
columns = pd.Index(columns)

# Home page route
@app.get("/")
async def get_home(request: Request):
    return templates.TemplateResponse("frontpage.html", {"request": request})

# Prediction page route
@app.get("/predict")
async def get_prediction_page(request: Request):
    return templates.TemplateResponse("prediction_page.html", {"request": request})

# Input schema
class HouseInput(BaseModel):
    location: str
    total_sqft: float
    area_type: str
    bhk: int
    bath: int

# Prediction function
def make_prediction(location, total_sqft, area_type, bedrooms, bathrooms):
    x_input = np.zeros(len(columns))
    x_input[0] = bedrooms
    x_input[1] = total_sqft
    x_input[2] = bathrooms

    if location in columns:
        x_input[columns.get_loc(location)] = 1

    if area_type in columns:
        x_input[columns.get_loc(area_type)] = 1

    x_df = pd.DataFrame([x_input], columns=columns)
    prediction_value = model.predict(x_df)[0] * 100000
    return prediction_value

# API route to get prediction
@app.post("/getprediction")
async def get_prediction(data: HouseInput):
    predicted_price = make_prediction(
        data.location, data.total_sqft, data.area_type, data.bhk, data.bath
    )
    return JSONResponse(content={"prediction": round(predicted_price, 2)})

# Run using Railway's PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
