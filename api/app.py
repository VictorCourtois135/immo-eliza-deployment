from fastapi import FastAPI
from schemas import PropertyInput, PredictionOutput
from predict import predict
app = FastAPI()

@app.get("/hello")
def say_hello():
    return {"message": "Hello!"}

@app.get("/")
def is_alive():
    return "alive"

@app.post("/predict")
def make_predict(new_property: PropertyInput):
    try: 
        new_property_dict = new_property.model_dump()
        new_prediction = predict(new_property_dict)
        return PredictionOutput(prediction= new_prediction)
    except Exception:
        print("!!! EXCEPTION CAUGHT !!!")
        return PredictionOutput(status_code=400)