import streamlit as st 
import requests
import json

URL_ML= "https://immo-eliza-deployment-xvsh.onrender.com"

st.header('House price prediction')

st.title('New house data:')

longitude = st.number_input('What is the logitude', format="%f")
latitude = st.number_input('What is the latitude', format="%f")
type_options = ['House', 'Apartment']
property_type = st.selectbox("Choose your property type:", type_options)
subtype_house_options = ['residence', 'mixed-building','bungalow','villa','cottage','chalet','master-house','mansion']
subtype_apartment_options = ['apartment', 'duplex', 'penthouse', 'studio', 'ground-floor', 'triplex','loft']

if property_type == 'House':
    property_subtype = st.selectbox("Choose your property sub type:", subtype_house_options)
elif property_type == 'Apartment':
    property_subtype = st.selectbox("Choose your property sub type:", subtype_apartment_options)
region_options= ['Wallonia', 'Brussels', 'Flanders']
region = st.selectbox('Name of the region', region_options)

province_wal_options=['Liège','Hainaut','Luxembourg','Namur','Walloon Brabant']
province_flem_options=['Limburg','East Flanders','Antwerp','West Flanders','Flemish Brabant']
province_bru_options=['Brussels Capital Region']

if  region == "Wallonia":
    province = st.selectbox('Name of the province',province_wal_options)
elif region == "Flanders":
    province = st.selectbox('Name of the province',province_flem_options)
elif region == "Brussels":
    province = st.selectbox('Name of the province',province_bru_options)
    
    
living_area_m2 = st.number_input('Number of living area in m²',step=1)
bedrooms = st.number_input('Number of bedrooms',step=1)
bathrooms = st.number_input('Number of bathrooms',step=1)
has_garden = st.checkbox('Is there a garden?')

if has_garden == True:
    garden_area_m2 = st.number_input('Area of the garden in m²',step=1)
else: 
    garden_area_m2 = None
    
building_year = st.number_input('Age of the building',step=1)

facades = st.number_input('Number od facades',step=1)
state_building= ['New','Normal', 'Excellent','To renovate','Fully renovated','To restore', 'Under construction','To demolish']

state_of_the_building = st.selectbox('What is the state of the building:', state_building)

epc_options=['A++','A+', 'A', 'B+','B','C', 'D','E+','E', 'F','G']

epc_score = st.selectbox('EPC score:', epc_options)

dict ={
    "latitude": latitude,
    "longitude": longitude,
    "property_type": property_type,
    "property_subtype": property_subtype,
    "region": region,
    "province": province,
    "living_area_m2": living_area_m2,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "facades": facades,
    "building_year": building_year,
    "garden_area_m2": garden_area_m2,
    "has_garden": has_garden,
    "state_of_the_building": state_of_the_building,
    "epc_score": epc_score
}

def get_predict(dict):
    url = f"{URL_ML}/predict"
    prediction = requests.post(url, json = dict)
    return prediction.json()["prediction"]


buton= st.button("Predict Price")

if buton:
    pred = st.write(get_predict(dict))
