from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import pickle

# Load trained model
with open("road_damage_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    road_age: float = Form(...),
    traffic_load: float = Form(...),
    rainfall: float = Form(...),
    soil_type: str = Form(...),
    pavement_type: str = Form(...),
    maintenance_freq: float = Form(...),
    temp_variation: float = Form(...)
):
    # Format input as a DataFrame
    input_data = pd.DataFrame([{
        "Road Age": road_age,
        "Traffic Load": traffic_load,
        "Rainfall": rainfall,
        "Soil Type": soil_type,
        "Pavement Type": pavement_type,
        "Maintenance Frequency": maintenance_freq,
        "Temperature Variation": temp_variation
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "YES" if prediction == 1 else "NO"

    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": result
    })