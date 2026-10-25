
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# MADRAS ESTATE | Chennai Property Intelligence
# =====================================================

st.set_page_config(
    page_title="MADRAS ESTATE | Chennai Property Intelligence",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = Path(__file__).parent / "Chennai houseing sale.csv"
MODEL_PATH = Path("model_tuned_candidate.pkl")
OUTPUT_DIR = Path("outputs")
MARKET_DATA_PATH = Path("market_data") / "Chennai-City HPI Data Current-Q (Base Year 2024).xls"

FOREST = "#244536"
DARK_GREEN = "#17382D"
BURGUNDY = "#7A1E2D"
IVORY = "#F6F1E8"
CREAM = "#FFF9F0"
BROWN = "#8B674B"
GOLD = "#C5A052"
SAGE = "#A4B39A"
TERRA = "#C56B4B"
INK = "#1E2923"
MUTED = "#716D64"

HERO_IMAGE = (
    "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c"
    "?auto=format&fit=crop&w=1800&q=85"
)

LOCALITY_IMAGES = {
    "Adyar": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=700&q=80",
    "Anna Nagar": "https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?auto=format&fit=crop&w=700&q=80",
    "T. Nagar": "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?auto=format&fit=crop&w=700&q=80",
    "Velachery": "https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?auto=format&fit=crop&w=700&q=80",
}


# -------------------- DATA AND MODEL --------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    if "AREA" in data.columns:
        data["AREA"] = data["AREA"].astype("string").str.strip()
        fixes = {
            "Adyr": "Adyar",
            "Ana Nagar": "Anna Nagar",
            "Ann Nagar": "Anna Nagar",
            "Chrompt": "Chromepet",
            "Chrmpet": "Chromepet",
            "TNagar": "T. Nagar",
            "T Nagar": "T. Nagar",
            "Velchery": "Velachery",
        }
        data["AREA"] = data["AREA"].replace(fixes)

    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_market_data():
    """Load the NHB RESIDEX Chennai assessment-price snapshot.

    The downloaded .xls file is an HTML table exported by the portal, so
    read_html is used. A transparent fallback keeps the app usable if the
    file parser is unavailable; update these values when a newer snapshot is downloaded.
    """
    fallback = pd.DataFrame({
        "Quarter": ["Jun 2025", "Sep 2025", "Dec 2025", "Mar 2026", "Jun 2026"],
        "Housing Price Index": [103.98, 103.72, 115.15, 111.00, 120.30],
        "Assessment price (₹/sq. ft.)": [9798, 9774, 10851, 10459, 11336],
    })

    if not MARKET_DATA_PATH.exists():
        return fallback, False

    try:
        tables = pd.read_html(MARKET_DATA_PATH, header=None)
        if not tables or tables[0].shape[1] < 3:
            return fallback, False
        raw = tables[0].copy()
        market = pd.DataFrame({
            "Quarter": raw.iloc[:, 0].astype(str).str.strip(),
            "Housing Price Index": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "Assessment price (₹/sq. ft.)": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
        }).dropna()
        if market.empty:
            return fallback, False
        return market.reset_index(drop=True), True
    except Exception:
        return fallback, False


df = load_data()
model = load_model()
market_df, market_file_loaded = load_market_data()


def money(value):
    return f"₹{float(value):,.0f}"


def options_for(column):
    return sorted(
        df[column].dropna().astype(str).str.strip().unique().tolist()
    )


def heading(kicker, title, description=""):
    st.markdown(
        f'<div class="kicker">{kicker}</div>'
        f'<h2 class="section-title">{title}</h2>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(
            f'<p class="description">{description}</p>',
            unsafe_allow_html=True,
        )


# -------------------- DESIGN --------------------

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {{
      --forest:{FOREST}; --darkgreen:{DARK_GREEN};
      --burgundy:{BURGUNDY}; --ivory:{IVORY};
      --cream:{CREAM}; --brown:{BROWN}; --gold:{GOLD};
      --sage:{SAGE}; --terra:{TERRA}; --ink:{INK};
      --muted:{MUTED};
    }}

    .stApp {{
      background:linear-gradient(145deg,#FFF9F0 0%,#F6F1E8 60%,#EFE2CF 100%);
      color:var(--ink);
      font-family:'Manrope',sans-serif;
    }}
    #MainMenu,footer,header {{visibility:hidden;}}
    [data-testid="stSidebar"] {{display:none;}}
    .block-container {{
      max-width:1450px;
      padding-top:1.2rem;
      padding-bottom:3rem;
    }}
    h1,h2,h3,h4 {{
      font-family:'Cormorant Garamond',Georgia,serif !important;
      color:var(--darkgreen) !important;
    }}
    .stApp p,.stApp label,.stApp li {{
      color:var(--ink);
      font-family:'Manrope',sans-serif;
    }}
    .brandbar {{
      display:flex;justify-content:space-between;align-items:center;
      gap:1rem;padding:.6rem 0 1rem;
      border-bottom:1px solid #E4D8C7;margin-bottom:1rem;
    }}
    .brand-name {{
      font-family:'Cormorant Garamond',serif;
      font-size:1.8rem;letter-spacing:.13em;color:var(--darkgreen);
    }}
    .brand-tag {{
      color:var(--brown);font-size:.65rem;
      letter-spacing:.17em;text-transform:uppercase;
    }}
    .kicker {{
      color:var(--terra) !important;
      font-size:.72rem;font-weight:800;
      letter-spacing:.18em;text-transform:uppercase;
      margin-top:.7rem;
    }}
    .section-title {{
      font-size:clamp(2rem,3vw,3rem);
      margin:.2rem 0;color:var(--darkgreen) !important;
    }}
    .description {{
      color:var(--muted) !important;
      margin-top:0;margin-bottom:1.2rem;
    }}
    .hero {{
      background:linear-gradient(100deg,rgba(23,56,45,.98),rgba(36,69,54,.88),rgba(36,69,54,.25)),
      url("{HERO_IMAGE}") center/cover;
      border:1px solid var(--gold);border-radius:20px;
      padding:clamp(1.5rem,4vw,4rem);
      min-height:330px;color:var(--cream);
    }}
    .hero h1 {{
      color:var(--cream) !important;
      font-size:clamp(3rem,5vw,5rem) !important;
      line-height:.95 !important;
      max-width:680px;
    }}
    .hero p {{color:#FFF4E3 !important;max-width:600px;line-height:1.8;}}
    .hero .eyebrow {{
      color:#E5C47D;font-size:.72rem;
      letter-spacing:.2em;text-transform:uppercase;
    }}
    .hero .highlight {{color:#E5B9A4;}}
    .stats {{
      display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
      gap:.8rem;margin:1rem 0 2rem;
    }}
    .stat {{
      background:rgba(255,249,240,.9);padding:1rem;
      border:1px solid #E4D8C7;border-top:3px solid var(--gold);
      border-radius:12px;
    }}
    .stat:nth-child(2) {{border-top-color:var(--burgundy);}}
    .stat:nth-child(3) {{border-top-color:var(--forest);}}
    .stat:nth-child(4) {{border-top-color:var(--brown);}}
    .stat-value {{
      font-family:'Cormorant Garamond',serif;
      font-size:2rem;color:var(--darkgreen);
    }}
    .stat-label {{font-size:.75rem;color:var(--muted);}}
    .price-panel {{
      background:linear-gradient(145deg,#7A1E2D,#591622);
      color:#FFF9F0;padding:1.5rem;border-radius:16px;
      border:1px solid var(--gold);
    }}
    .price-panel h2,.price-panel h3 {{
      color:#FFF4E3 !important;
    }}
    .price-panel p,.price-panel div {{
      color:#FFF4E3;
    }}
    .price-label {{
      color:#E8CFA0 !important;font-size:.72rem;
      letter-spacing:.15em;text-transform:uppercase;
    }}
    .price-value {{
      font-family:'Cormorant Garamond',serif;
      color:#FFF9F0 !important;font-size:3rem;
    }}
    .locality-title {{
      font-family:'Cormorant Garamond',serif;
      font-size:1.6rem;color:var(--darkgreen);
    }}
    .locality-price {{color:var(--burgundy);font-weight:800;}}
    .small-note {{color:var(--muted);font-size:.78rem;}}
    .callout {{
      padding:1rem;border-left:4px solid var(--terra);
      border-radius:0 10px 10px 0;
      background:rgba(197,160,82,.12);
      color:var(--ink);line-height:1.7;
    }}
    .footer-note {{
      border-top:1px solid #E4D8C7;padding-top:1rem;
      margin-top:2rem;color:var(--muted);font-size:.75rem;
    }}

    /* Dark text on light navigation buttons */
    div[role="radiogroup"] {{gap:.4rem;}}
    div[role="radiogroup"] label {{
      background:#FFF9F0;border:1px solid #E4D8C7;
      padding:.45rem .8rem;border-radius:999px;
    }}
    div[role="radiogroup"] label p {{
      color:var(--ink) !important;font-weight:700;font-size:.8rem;
    }}
    div[role="radiogroup"] label:has(input:checked) {{
      background:var(--forest);border-color:var(--forest);
    }}
    div[role="radiogroup"] label:has(input:checked) p {{
      color:#FFF9F0 !important;
    }}

    /* Buttons */
    .stButton>button,.stFormSubmitButton>button {{
      background:var(--burgundy);color:#FFF9F0;
      border:1px solid var(--burgundy);border-radius:9px;
      font-weight:700;padding:.65rem 1rem;
    }}
    .stButton>button:hover,.stFormSubmitButton>button:hover {{
      background:var(--forest);border-color:var(--forest);
      color:#FFF9F0;
    }}
    [data-testid="stMetric"] {{
      background:#FFF9F0;border:1px solid #E4D8C7;
      border-radius:12px;padding:.8rem;
    }}
    [data-testid="stMetricLabel"] {{color:var(--muted) !important;}}
    [data-testid="stMetricValue"] {{
      color:var(--darkgreen) !important;
      font-family:'Cormorant Garamond',serif;
    }}
    [data-testid="stImage"] img {{
      width:100%;height:145px;object-fit:cover;border-radius:9px;
    }}
    .stAlert {{border-radius:10px;}}

    @media(max-width:800px) {{
      .stats {{grid-template-columns:repeat(2,minmax(0,1fr));}}
      .brand-name {{font-size:1.3rem;}}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- BRAND AND NAVIGATION --------------------

st.markdown(
    """
    <div class="brandbar">
      <div>
        <div class="brand-name">⌂ MADRAS ESTATE</div>
        <div class="brand-tag">Chennai Property Intelligence</div>
      </div>
      <div class="brand-tag">A brighter view of Chennai living</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Homepage button requests a page change on the next rerun.
if st.session_state.pop("go_to_estimator", False):
    st.session_state["page_nav"] = "Price Estimator"

page = st.radio(
    "Main navigation",
    ["Home", "Price Estimator", "Explore Chennai", "Chennai Market Trends", "About & Methodology"],
    horizontal=True,
    label_visibility="collapsed",
    key="page_nav",
)

st.markdown("---")


# -------------------- HOME --------------------

if page == "Home":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">A brighter view of Chennai living</div>
          <h1>Find a place<br>to call <span class="highlight">yours.</span></h1>
          <p>Explore Chennai neighbourhoods, follow recent housing-market trends, and estimate a property's price.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    area_medians = df.groupby("AREA")["SALES_PRICE"].median().sort_values(ascending=False)

    st.markdown(
        f"""
        <div class="stats">
          <div class="stat"><div class="stat-value">{len(df):,}</div><div class="stat-label">Historical property records</div></div>
          <div class="stat"><div class="stat-value">16</div><div class="stat-label">Property input features</div></div>
          <div class="stat"><div class="stat-value">{df['AREA'].nunique()}+</div><div class="stat-label">Neighbourhoods covered</div></div>
          <div class="stat"><div class="stat-value">ML</div><div class="stat-label">Random Forest estimate</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, .85], gap="large")

    with left:
        heading(
            "Explore the city",
            "Neighbourhoods with character",
            "Compare past sale-price patterns across Chennai neighbourhoods.",
        )

        preferred = ["Adyar", "Anna Nagar", "T. Nagar", "Velachery"]
        names = [name for name in preferred if name in area_medians.index]
        for name in area_medians.index:
            if name not in names and len(names) < 4:
                names.append(name)

        card_cols = st.columns(2, gap="medium")
        for i, name in enumerate(names):
            with card_cols[i % 2]:
                with st.container(border=True):
                    if name in LOCALITY_IMAGES:
                        st.image(LOCALITY_IMAGES[name], use_container_width=True)
                    st.markdown(
                        f'<div class="locality-title">{name}</div>'
                        f'<div class="locality-price">{money(area_medians[name])}</div>'
                        '<div class="small-note">Historical median sale price</div>',
                        unsafe_allow_html=True,
                    )

    with right:
        st.markdown(
            """
            <div class="price-panel">
              <div class="price-label">Your next step</div>
              <h2>A clearer picture starts here.</h2>
              <p>Explore a property's estimated sale price using its locality, floor area, building type and quality scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button(
            "Estimate a property price  →",
            use_container_width=True,
            key="home_estimate_cta",
        ):
            st.session_state["go_to_estimator"] = True
            st.rerun()

        heading("At a glance", "The data behind the estimate")
        highest_area = area_medians.index[0]
        st.markdown(
            f"""
            <div class="callout">
              Property records cover several Chennai neighbourhoods. The highest historical median
              is in <b>{highest_area}</b> at <b>{money(area_medians.iloc[0])}</b>.
              These are historical summaries, not live market quotes.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------- PRICE ESTIMATOR --------------------

elif page == "Price Estimator":
    heading(
        "Property intelligence",
        "Estimate a property price",
        "Answer each question about the property. Plain-language explanations are provided throughout.",
    )

    # Display friendly labels while passing model-compatible categories.
    aliases = {
        "AREA": {"Adyr": "Adyar", "Ana Nagar": "Anna Nagar", "Ann Nagar": "Anna Nagar", "Chrompt": "Chrompet", "Chrmpet": "Chrompet", "Chormpet": "Chrompet", "Velchery": "Velachery", "Kknagar": "KK Nagar", "KKNagar": "KK Nagar", "Karapakam": "Karapakkam", "TNagar": "T Nagar", "T. Nagar": "T Nagar"},
        "BUILDTYPE": {"Comercial": "Commercial", "Other": "Others"},
        "SALE_COND": {"Ab Normal": "AbNormal", "Adj Land": "AdjLand", "Partiall": "Partial", "PartiaLl": "Partial"},
        "PARK_FACIL": {"Noo": "No"},
        "UTILITY_AVAIL": {"All Pub": "AllPub"},
        "STREET": {"Pavd": "Paved", "NoAccess": "No Access"},
    }

    def canonical(column, value):
        raw = str(value).strip()
        return aliases.get(column, {}).get(raw, raw)

    def clean_options(column):
        return sorted({canonical(column, value) for value in df[column].dropna().astype(str)})

    def common_value(column, options):
        values = df[column].dropna().astype(str).map(lambda value: canonical(column, value))
        if values.empty or not options:
            return options[0] if options else None
        return values.value_counts().index[0]

    def median_value(column, fallback):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else fallback

    labels = {
        "BUILDTYPE": {
            "House": "Residential house",
            "Commercial": "Commercial building (mainly used for business)",
            "Others": "Other building type",
        },
        "SALE_COND": {
            "Normal Sale": "Standard property sale",
            "AbNormal": "Sale under unusual circumstances",
            "AdjLand": "Sale involving adjoining land",
            "Family": "Sale between family members",
            "Partial": "Property was incomplete when assessed",
        },
        "UTILITY_AVAIL": {
            "AllPub": "Electricity, water and sewer connection available",
            "NoSewr": "No sewer connection",
            "NoSeWa": "Water and sewer facilities unavailable",
            "ELO": "Electricity only",
        },
        "STREET": {
            "Paved": "Paved road surface",
            "Gravel": "Gravel road surface",
            "No Access": "No direct road access recorded",
        },
        "MZZONE": {
            "RL": "Residential zone — low density",
            "RM": "Residential zone — medium density",
            "RH": "Residential zone — high density",
            "C": "Commercial zone",
            "I": "Industrial zone",
            "A": "Agricultural zone",
        },
    }

    def pretty(column, value):
        value = canonical(column, value)
        if column == "PARK_FACIL":
            return "Yes — parking is available" if value == "Yes" else "No — parking is not available"
        return labels.get(column, {}).get(value, value)

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        with st.form("prediction_form"):
            st.subheader("1. Property details")
            st.caption("All fields are required for this estimate.")

            area_options = clean_options("AREA")
            area = st.selectbox(
                "Which Chennai neighbourhood is the property in?",
                area_options,
                format_func=lambda value: "T. Nagar" if value == "T Nagar" else value,
                help="Choose the neighbourhood where the property is located.",
            )
            sqft = st.number_input(
                "What is the built-up area? (square feet)", min_value=300, max_value=10000,
                value=1000, step=50, format="%d",
                help="Built-up area is the constructed floor area, not the land-plot size. An approximate figure is fine.",
            )
            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.selectbox("How many bedrooms are there?", [1, 2, 3, 4, 5, 6], index=1,
                                        help="Count bedrooms only—not the living room or kitchen.")
                bathrooms = st.selectbox("How many bathrooms/toilets are there?", [1, 2, 3, 4], index=0,
                                         help="Count the bathrooms and toilets in the property.")
            with c2:
                rooms = st.selectbox("How many rooms are there in total?", [2, 3, 4, 5, 6, 7, 8], index=2,
                                     help="Count bedrooms, living/dining rooms and other rooms. Do not count bathrooms as rooms.")
                building_options = clean_options("BUILDTYPE")
                building_default = common_value("BUILDTYPE", building_options)
                buildtype = st.selectbox("What is the building mainly used for?", building_options,
                                         index=building_options.index(building_default),
                                         format_func=lambda value: pretty("BUILDTYPE", value),
                                         help="Choose residential house, commercial building, or other building type.")

            parking = st.radio("Is a parking space available with the property?", ["Yes", "No"], horizontal=True,
                                format_func=lambda value: "Yes — parking is available" if value == "Yes" else "No — parking is not available",
                                help="Choose whether the property includes a parking facility.")

            st.subheader("2. Access and facilities")
            distance_default = int(round(median_value("DIST_MAINROAD", 100)))
            distance = st.number_input("Approximately how far is the property from the main road? (metres)",
                                       min_value=0, max_value=5000, value=max(0, min(5000, distance_default)), step=10,
                                       help="Enter an approximate distance from the main road. Use your best estimate if you do not know the exact distance.")
            c1, c2 = st.columns(2)
            with c1:
                utility_options = clean_options("UTILITY_AVAIL")
                utility_default = common_value("UTILITY_AVAIL", utility_options)
                utility = st.selectbox("Which essential utilities are available?", utility_options,
                                       index=utility_options.index(utility_default),
                                       format_func=lambda value: pretty("UTILITY_AVAIL", value),
                                       help="Choose the description that best matches electricity, water and sewer facilities at the property.")
            with c2:
                street_options = clean_options("STREET")
                street_default = common_value("STREET", street_options)
                street = st.selectbox("What is the road access like?", street_options,
                                      index=street_options.index(street_default),
                                      format_func=lambda value: pretty("STREET", value),
                                      help="Choose the road surface or access description that best matches the property.")

            zone_options = clean_options("MZZONE")
            zone_default = common_value("MZZONE", zone_options)
            zone = st.selectbox("What is the property's land-use zone?", zone_options,
                                index=zone_options.index(zone_default),
                                format_func=lambda value: pretty("MZZONE", value),
                                help="This is the planning designation for the area. Check property documents or ask the owner/agent if you are unsure.")

            st.subheader("3. Sale circumstances")
            sale_options = clean_options("SALE_COND")
            sale_default = common_value("SALE_COND", sale_options)
            sale_cond = st.selectbox("Which description best matches the sale?", sale_options,
                                     index=sale_options.index(sale_default),
                                     format_func=lambda value: pretty("SALE_COND", value),
                                     help="This describes the circumstances of the sale, not the building. For an ordinary sale, choose 'Standard property sale'.")

            st.subheader("4. Rate the property's condition and finish")
            st.caption("Use the same guide for each rating. These are your estimates, not an official inspection.")
            st.info("2 = basic · 3 = average · 4 = good · 5 = excellent")
            quality = {}
            quality_fields = [
                ("QS_ROOMS", "How would you rate the finish of the rooms?", "Consider flooring, walls, doors and general finish."),
                ("QS_BATHROOM", "How would you rate the bathroom finish?", "Consider fittings, tiling and overall condition."),
                ("QS_BEDROOM", "How would you rate the bedroom finish?", "Consider flooring, walls, storage and general condition."),
                ("QS_OVERALL", "Overall, how would you rate the property?", "Consider the property's general condition and finish as a whole."),
            ]
            for i in range(0, len(quality_fields), 2):
                qa, qb = st.columns(2)
                for target, item in zip((qa, qb), quality_fields[i:i + 2]):
                    column, label, explanation = item
                    with target:
                        quality[column] = st.slider(
                            label, min_value=2.0, max_value=5.0,
                            value=float(min(5.0, max(2.0, round(median_value(column, 3.5), 1)))),
                            step=0.1, help=explanation + " Use 2 for basic, 3 for average, 4 for good and 5 for excellent.",
                        )
            submitted = st.form_submit_button("Estimate the price  →", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="price-panel">
              <div class="price-label">Before you begin</div>
              <h2>A clearer estimate starts with clear details.</h2>
              <p>Answer each question using the information available to you. For planning-zone or sale-circumstance details, check the property documents or ask the owner or agent if needed.</p>
              <p>The result is an indicative estimate, not a guaranteed selling price or an official valuation.</p>
            </div>
            """, unsafe_allow_html=True,
        )
        st.write("")
        with st.container(border=True):
            st.markdown("**Quick guide**")
            st.markdown(
                "- **Built-up area:** constructed floor area, measured in square feet.\n"
                "- **Land-use zone:** the planning category assigned to an area.\n"
                "- **Utilities:** services such as electricity, water and sewerage.\n"
                "- **Sale circumstances:** the situation under which the property is sold.\n"
                "- **Condition rating:** your estimate of visible finish and condition."
            )

    if submitted:
        input_data = pd.DataFrame([{
            "AREA": area, "INT_SQFT": sqft, "DIST_MAINROAD": distance,
            "N_BEDROOM": bedrooms, "N_BATHROOM": bathrooms, "N_ROOM": rooms,
            "SALE_COND": sale_cond, "PARK_FACIL": parking, "BUILDTYPE": buildtype,
            "UTILITY_AVAIL": utility, "STREET": street, "MZZONE": zone,
            "QS_ROOMS": quality["QS_ROOMS"], "QS_BATHROOM": quality["QS_BATHROOM"],
            "QS_BEDROOM": quality["QS_BEDROOM"], "QS_OVERALL": quality["QS_OVERALL"],
        }])
        try:
            predicted_price = float(model.predict(input_data)[0])
            st.markdown(
                f"""
                <div class="price-panel">
                  <div class="price-label">Estimated property price</div>
                  <div class="price-value">{money(predicted_price)}</div>
                  <p>This is an indicative estimate based on the information entered. Actual selling prices can differ.</p>
                </div>
                """, unsafe_allow_html=True,
            )
            st.caption("The prediction model was trained on older property records and has not been validated against recent individual-property sales. Treat this result as an estimate, not a verified current-market price.")
        except Exception as error:
            st.error(f"We couldn't calculate the estimate. Please check the property details and try again. Technical detail: {error}")


# -------------------- EXPLORE CHENNAI --------------------

elif page == "Explore Chennai":
    heading(
        "Locality intelligence",
        "Explore Chennai",
        "Compare historical sale prices in the locality categories available in the available property information.",
    )

    stats = df.groupby("AREA")["SALES_PRICE"].agg(
        ["median", "mean", "count"]
    ).sort_values("median", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Neighbourhoods covered", len(stats))
    c2.metric("Highest historical median", money(stats["median"].max()))
    c3.metric("Property records", f"{int(stats['count'].sum()):,}")

    st.markdown("---")
    left, right = st.columns([1.1, .9], gap="large")

    with left:
        heading("Historical comparison", "Median sale price by locality")
        fig, ax = plt.subplots(figsize=(9, 6))
        values = stats["median"].sort_values()
        colors = [SAGE, GOLD, BROWN, BURGUNDY, FOREST]
        ax.barh(
            values.index.astype(str),
            values.values,
            color=[colors[i % len(colors)] for i in range(len(values))],
        )
        ax.set_xlabel("Median historical sale price (₹)")
        ax.set_ylabel("")
        ax.set_facecolor(IVORY)
        fig.patch.set_facecolor(IVORY)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#D8CBB8")
        ax.tick_params(axis="y", labelsize=8, length=0)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        heading("Side by side", "Compare localities")
        localities = sorted(stats.index.astype(str).tolist())
        default = [
            x for x in ["Adyar", "Anna Nagar", "T. Nagar", "Velachery"]
            if x in localities
        ][:2]
        selected = st.multiselect(
            "Choose localities", localities, default=default
        )

        if selected:
            comparison = stats.loc[selected].reset_index()
            comparison.columns = [
                "Locality", "Median price", "Mean price", "Record count"
            ]
            comparison["Median price"] = comparison["Median price"].map(money)
            comparison["Mean price"] = comparison["Mean price"].map(money)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        else:
            st.info("Choose one or more localities to compare.")

    st.markdown(
        """
        <div class="callout">
          These figures summarize past property sales. They are not
          live listings or verified current market prices.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- CHENNAI MARKET TRENDS --------------------

elif page == "Chennai Market Trends":
    heading(
        "Chennai housing market",
        "Market trends at a glance",
        "Quarterly housing-price index and assessment-price indicators reported for Chennai.",
    )

    latest = market_df.iloc[-1]
    first = market_df.iloc[0]
    change_pct = ((float(latest["Assessment price (₹/sq. ft.)"]) / float(first["Assessment price (₹/sq. ft.)"])) - 1) * 100
    index_change_pct = ((float(latest["Housing Price Index"]) / float(first["Housing Price Index"])) - 1) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest reported quarter", str(latest["Quarter"]))
    m2.metric("Assessment price per sq. ft.", f"₹{float(latest['Assessment price (₹/sq. ft.)']):,.0f}")
    m3.metric("Change across displayed period", f"{change_pct:+.1f}%", help="Change from the first to the last quarter shown here; it is not a forecast.")

    st.markdown("---")
    left, right = st.columns(2, gap="large")
    quarter_labels = market_df["Quarter"].astype(str).tolist()
    x = list(range(len(market_df)))
    price_values = pd.to_numeric(market_df["Assessment price (₹/sq. ft.)"], errors="coerce").tolist()
    index_values = pd.to_numeric(market_df["Housing Price Index"], errors="coerce").tolist()

    with left:
        heading("Price indicator", "Assessment price per square foot")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(x, price_values, marker="o", markersize=6, linewidth=2.5, color=BURGUNDY, zorder=3)
        ax.fill_between(x, price_values, [min(price_values) * 0.97] * len(price_values), alpha=0.10, color=BURGUNDY)
        ax.set_xticks(x, quarter_labels)
        ax.set_ylim(min(price_values) * 0.94, max(price_values) * 1.06)
        ax.set_ylabel("₹ per sq. ft.")
        ax.set_xlabel("")
        ax.set_facecolor(IVORY)
        fig.patch.set_facecolor(IVORY)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#D8CBB8")
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(axis="x", rotation=25, length=0, labelsize=9)
        ax.margins(x=0.08)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        heading("Market index", "Housing Price Index")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(x, index_values, marker="o", markersize=6, linewidth=2.5, color=FOREST, zorder=3)
        ax.set_xticks(x, quarter_labels)
        ax.set_ylim(min(index_values) - 2, max(index_values) + 2)
        ax.set_ylabel("Index value")
        ax.set_xlabel("")
        ax.set_facecolor(IVORY)
        fig.patch.set_facecolor(IVORY)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#D8CBB8")
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(axis="x", rotation=25, length=0, labelsize=9)
        ax.margins(x=0.08)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    heading("Quarterly figures", "View the reported values")
    display_market = market_df.copy()
    display_market["Assessment price (₹/sq. ft.)"] = display_market["Assessment price (₹/sq. ft.)"].map(lambda value: f"₹{float(value):,.0f}")
    display_market["Housing Price Index"] = display_market["Housing Price Index"].map(lambda value: f"{float(value):.2f}")
    st.dataframe(display_market, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="callout">
          <b>How to read this:</b> These are city-level housing-market indicators, not the confirmed selling price of a specific home. Individual prices vary by neighbourhood, size, condition and other factors. These figures do not automatically adjust the price estimate produced by the property estimator.
        </div>
        """,
        unsafe_allow_html=True,
    )
    source_status = "Loaded from the NHB RESIDEX file in your market_data folder." if market_file_loaded else "Showing the values from the NHB RESIDEX Current-Q snapshot you supplied. If you update the file later, keep the same filename to load the new values."
    st.caption(f"Source: NHB RESIDEX · Chennai · HPI@Assessment Price · Carpet area prices (₹/sq. ft.) · Base year 2024. {source_status}")


# -------------------- ABOUT AND METHODOLOGY --------------------

elif page == "About & Methodology":
    heading(
        "Behind the estimate",
        "About MADRAS ESTATE",
        "Explore Chennai neighbourhoods and receive an indicative property-price estimate.",
    )

    st.markdown(
        """
        <div class="callout">
          MADRAS ESTATE combines property exploration with a supervised
          regression model. It is intended for learning and demonstration,
          not as a substitute for professional property valuation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    a, b, c = st.columns(3)

    with a:
        st.subheader("01 · Prepare the data")
        st.write(
            "Inspect historical records, clean category text, handle missing "
            "values and encode categorical features."
        )

    with b:
        st.subheader("02 · Train and compare")
        st.write(
            "Compare Linear Regression, Decision Tree and Random Forest "
            "using a held-out test set."
        )

    with c:
        st.subheader("03 · Evaluate responsibly")
        st.write(
            "Use R², MAE and RMSE to understand test performance. "
            "Good metrics do not guarantee current-market accuracy."
        )

    st.markdown("---")
    heading(
        "Model insights",
        "Training visualisations",
        "These charts explain how the prediction model was evaluated.",
    )

    graph_files = [
        ("Model comparison", "model_comparison.png"),
        ("Actual vs predicted", "actual_vs_predicted.png"),
        ("Residual plot", "residual_plot.png"),
        ("Feature importance", "feature_importance.png"),
    ]

    for start in range(0, len(graph_files), 2):
        cols = st.columns(2)
        for col, (title, filename) in zip(cols, graph_files[start:start + 2]):
            with col:
                st.subheader(title)
                image_path = OUTPUT_DIR / filename
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning(
                        f"{filename} not found. Run train_model.py to generate it."
                    )

    st.markdown("---")
    heading("Technology", "Built with practical tools")
    st.write(
        "**Python** · **Pandas** · **Scikit-learn** · **Matplotlib** · **Streamlit**"
    )

    st.markdown(
        """
        <div class="callout">
          <b>Limitations:</b> predictions depend on the property records used to train the model, input
          quality and model assumptions. The data is historical and does not
          guarantee present-day accuracy. Feature importance indicates model
          association, not causation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- FOOTER --------------------

st.markdown(
    """
    <div class="footer-note">
      MADRAS ESTATE · Chennai Property Intelligence · Educational use only.
      Historical estimates are not official valuations.
    </div>
    """,
    unsafe_allow_html=True,
)
