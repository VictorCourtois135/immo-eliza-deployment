import streamlit as st
import requests
import os
from dotenv import load_dotenv
from urllib.parse import quote

# --- Load environment variables ---

load_dotenv()

URL_ML = os.getenv("URL_ML")
API_KEY_GEO = os.getenv("API_KEY_GEO")

if not URL_ML or not API_KEY_GEO:
    st.error("Missing environment variables. Check your .env file.")
    st.stop()


# =========================
# CONSTANTS
# =========================
PROPERTY_TYPES = ['House', 'Apartment']
SUBTYPE_HOUSE = ['residence', 'mixed-building', 'bungalow', 'villa', 'cottage', 'chalet', 'master-house', 'mansion']
SUBTYPE_APARTMENT = ['apartment', 'duplex', 'penthouse', 'studio', 'ground-floor', 'triplex', 'loft']

PROVINCES = {
    'Wallonia': ['Liège', 'Hainaut', 'Luxembourg', 'Namur', 'Walloon Brabant'],
    'Flanders': ['Limburg', 'East Flanders', 'Antwerp', 'West Flanders', 'Flemish Brabant'],
    'Brussels': ['Brussels Capital Region'],
}
# Reverse lookup: province -> region, used as a fallback when only the
# province could be detected from the geocoding result.
PROVINCE_TO_REGION = {province: region for region, provinces in PROVINCES.items() for province in provinces}

STATE_OPTIONS = ['New', 'Under construction', 'Fully renovated', 'Excellent', 'Normal', 'To renovate', 'To restore', 'To demolish']
EPC_OPTIONS = ['A++', 'A+', 'A', 'B+', 'B', 'C', 'D', 'E+', 'E', 'F', 'G']

EPC_COLORS = {
    'A++': '#1E7A4A', 'A+': '#2E8B57', 'A': '#4CA35F',
    'B+': '#8FAE3D', 'B': '#B7B93A',
    'C': '#D9A62E', 'D': '#D98A2E',
    'E+': '#D9702E', 'E': '#C9562A',
    'F': '#B33A2A', 'G': '#8C1F1F',
}

# Belgian postal codes map directly and unambiguously to a province — unlike
# Geoapify's "state"/"county" text fields, which vary by language (FR/NL/EN)
# and sometimes report the arrondissement instead of the province (e.g. an
# address in Verviers returns county="Verviers", not "Liège"). Postcodes are
# just numbers, always present, and never ambiguous — so this single table
# replaces what used to be ~90 lines of multilingual aliases.
POSTCODE_RANGES = [
    (1000, 1299, 'Brussels Capital Region'),
    (1300, 1499, 'Walloon Brabant'),
    (1500, 1999, 'Flemish Brabant'),
    (2000, 2999, 'Antwerp'),
    (3000, 3499, 'Flemish Brabant'),
    (3500, 3999, 'Limburg'),
    (4000, 4999, 'Liège'),
    (5000, 5999, 'Namur'),
    (6000, 6599, 'Hainaut'),
    (6600, 6999, 'Luxembourg'),
    (7000, 7999, 'Hainaut'),
    (8000, 8999, 'West Flanders'),
    (9000, 9999, 'East Flanders'),
]


def detect_region_province(postcode):
    """Map a Belgian postal code to (region, province).
    Returns (None, None) if the postcode is missing or out of the known ranges."""
    try:
        code = int(postcode)
    except (TypeError, ValueError):
        return None, None

    for low, high, province in POSTCODE_RANGES:
        if low <= code <= high:
            return PROVINCE_TO_REGION[province], province
    return None, None


# =========================
# FUNCTIONS
# =========================
@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    """Geocode an address via Geoapify. Returns (result_dict, error_message)."""
    encoded_address = quote(address)
    url = f"https://api.geoapify.com/v1/geocode/search?text={encoded_address}&apiKey={API_KEY_GEO}"
    resp = requests.get(url, headers={"Accept": "application/json"})

    if resp.status_code != 200:
        return None, f"Geoapify API error: {resp.status_code}"

    features = resp.json().get("features")
    if not features:
        return None, "No results found for this address"

    props = features[0]["properties"]
    lon, lat = features[0]["geometry"]["coordinates"]

    result = {
        "lon": lon,
        "lat": lat,
        "postcode": props.get("postcode"),
    }
    return result, None


def get_predict(data: dict):
    resp = requests.post(f"{URL_ML}/predict", json=data)
    resp.raise_for_status()
    return resp.json()  # {"prediction": ..., "low": ..., "high": ..., "status_code": ...}


# =========================
# PAGE CONFIG & STYLE
# =========================
st.set_page_config(page_title="Property Valuation", page_icon="📐", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #FAF6EE;
}

/* Top header bar (☰ menu / Deploy button) — transparent so it shows
   whatever is behind it, staying in sync with .stApp automatically */
header[data-testid="stHeader"] {
    background: transparent !important;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
    color: #17293D;
}

p, label, span, div {
    color: #3C4A57;
}

/* Hero */
.hero {
    background:
        repeating-linear-gradient(0deg, rgba(23,41,61,0.05) 0px, rgba(23,41,61,0.05) 1px, transparent 1px, transparent 32px),
        repeating-linear-gradient(90deg, rgba(23,41,61,0.05) 0px, rgba(23,41,61,0.05) 1px, transparent 1px, transparent 32px);
    border: 1px solid #D9CFB8;
    border-radius: 4px;
    padding: 2.2rem 2rem 1.8rem 2rem;
    margin-bottom: 1.8rem;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: #A9762F;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: #17293D;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #5C6975;
    max-width: 34rem;
}

/* Section eyebrow labels (FIG. 0X) */
.fig-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A9762F;
    border-bottom: 1px dashed #A9762F;
    display: inline-block;
    padding-bottom: 3px;
    margin-bottom: 0.1rem;
}
.fig-title {
    font-family: 'Fraunces', serif;
    font-size: 1.25rem;
    color: #17293D;
    margin-top: 0.15rem;
    margin-bottom: 0.9rem;
}

/* Bordered container used for each section card */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E3DAC3 !important;
    border-radius: 6px !important;
    background-color: #FFFFFF;
}

/* Buttons */
.stButton>button {
    background-color: #3F6B63;
    border: 1px solid #3F6B63;
    border-radius: 4px;
    padding: 0.65rem 1.4rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.03em;
    width: 100%;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}

/* Force the button's inner text (p/span/div) to stay light,
   overriding the global "p, span, div" color rule above */
.stButton>button,
.stButton>button p,
.stButton>button span,
.stButton>button div {
    color: #FAF6EE !important;
}

.stButton>button:hover {
    background-color: #305148;
    border-color: #305148;
}
.stButton>button:active {
    background-color: #24413B;
    border-color: #24413B;
}
.stButton>button:focus:not(:active) {
    background-color: #3F6B63;
    border-color: #A9762F;
}

/* Result stamp */
.stamp {
    border: 1.5px dashed #A9762F;
    border-radius: 6px;
    padding: 1.6rem 1.8rem;
    background-color: #FFFDF8;
    text-align: center;
    margin-top: 1rem;
}
.stamp-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A9762F;
    margin-bottom: 0.4rem;
}
.stamp-value {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #17293D;
}
.stamp-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: #3F6B63;
    margin-top: 0.5rem;
}
.stamp-range-caption {
    font-size: 0.75rem;
    color: #8A8370;
    margin-top: 0.2rem;
}

/* EPC badge */
.epc-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    color: #FFFFFF;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    margin-left: 0.5rem;
}

hr {
    border-top: 1px dashed #D9CFB8;
}
</style>
""", unsafe_allow_html=True)


# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Fig. 00 — Estimation form</div>
    <div class="hero-title">What is your property worth?</div>
    <div class="hero-subtitle">
        Enter the address and a few characteristics of the property.
        We locate it, and our model returns an estimated market value in seconds.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# SECTION 01 — LOCATION & TYPE
# =========================
st.markdown('<div class="fig-label">Fig. 01</div><div class="fig-title">Location & type</div>', unsafe_allow_html=True)

with st.container(border=True):
    address = st.text_input("Property address", placeholder="e.g. Rue de la Loi 16, Brussels")

    geo, geo_error = (geocode_address(address) if address.strip() else (None, None))

    region, province = None, None
    if geo:
        region, province = detect_region_province(geo["postcode"])

    col1, col2 = st.columns(2)
    with col1:
        property_type = st.selectbox("Property type", PROPERTY_TYPES)
    with col2:
        subtype_options = SUBTYPE_HOUSE if property_type == 'House' else SUBTYPE_APARTMENT
        property_subtype = st.selectbox("Property subtype", subtype_options)

    if address.strip():
        if region:
            st.caption(f"📍 {region} — {province or 'province unknown'}")
        elif geo:
            st.caption("⚠️ Couldn't determine the region/province for this address. Try adding the city name.")
        elif geo_error:
            st.caption(f"⚠️ {geo_error}")

        if geo and not region:
            with st.expander("Debug: raw Geoapify postcode"):
                st.write(geo["postcode"])

st.write("")


# =========================
# SECTION 02 — CHARACTERISTICS
# =========================
st.markdown('<div class="fig-label">Fig. 02</div><div class="fig-title">Characteristics</div>', unsafe_allow_html=True)

with st.container(border=True):
    col5, col6, col7 = st.columns(3)
    with col5:
        living_area_m2 = st.number_input("Living area (m²)", min_value=0, step=1)
    with col6:
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    with col7:
        bathrooms = st.number_input("Bathrooms", min_value=0, step=1)

    col8, col9 = st.columns(2)
    with col8:
        building_year = st.number_input("Building year", min_value=1800, max_value=2026, step=1, value=2000)
    with col9:
        facades = st.number_input("Number of facades", min_value=0, max_value=4, step=1)

    has_garden = st.checkbox("Has a garden")
    garden_area_m2 = st.number_input("Garden area (m²)", min_value=0, step=1) if has_garden else None

st.write("")


# =========================
# SECTION 03 — CONDITION & ENERGY
# =========================
st.markdown('<div class="fig-label">Fig. 03</div><div class="fig-title">Condition & energy</div>', unsafe_allow_html=True)

with st.container(border=True):
    col10, col11 = st.columns(2)
    with col10:
        state_of_the_building = st.selectbox("Building condition", STATE_OPTIONS)
    with col11:
        epc_score = st.selectbox("EPC score", EPC_OPTIONS)
        badge_color = EPC_COLORS.get(epc_score, "#3C4A57")
        st.markdown(
            f'<span class="epc-badge" style="background-color:{badge_color};">EPC {epc_score}</span>',
            unsafe_allow_html=True
        )

st.write("")


# =========================
# SUBMIT
# =========================
submitted = st.button("📐  Estimate the price")

if submitted:
    if not address.strip():
        st.warning("Please enter an address.")
    elif geo_error:
        st.warning(geo_error)
    elif not region or not province:
        st.warning("Couldn't determine the region/province for this address. Please make it more specific (e.g. add the city name) and try again.")
    else:
        longitude, latitude = geo["lon"], geo["lat"]

        data_dict = {
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
            "epc_score": epc_score,
        }

        with st.spinner(text="Estimating price...", show_time=True):
            try:
                result = get_predict(data_dict)
                pred = result.get("prediction")
                low = result.get("low")
                high = result.get("high")

                if low is not None and high is not None:
                    range_html = f"""
                    <div class="stamp-range">€ {low:,.0f} — € {high:,.0f}</div>
                    <div class="stamp-range-caption">Typical price range (± average model error)</div>
                    """.replace(",", " ")
                else:
                    range_html = ""

                st.markdown(f"""
                <div class="stamp">
                    <div class="stamp-label">Fig. 04 — Estimated market value</div>
                    <div class="stamp-value">€ {pred:,.0f}</div>
                    {range_html}
                </div>
                """.replace(",", " "), unsafe_allow_html=True)
            except requests.exceptions.RequestException as e:
                st.error(f"Prediction API error: {e}")