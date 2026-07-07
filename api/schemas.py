from typing import Optional
from pydantic import BaseModel

class PropertyInput(BaseModel):
    latitude: float
    longitude: float
    property_type: str
    property_subtype: str
    region: str
    province: str
    living_area_m2: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    facades: Optional[int] = None
    building_year: Optional[int] = None
    garden_area_m2: Optional[int] = None
    has_garden: Optional[bool] = False
    state_of_the_building: Optional[str] = None
    epc_score: Optional[str] = None 
    
    
class PredictionOutput(BaseModel):
    prediction: Optional[float] = None
    status_code: Optional[int] = 200