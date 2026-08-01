import streamlit as st
from streamlit_option_menu import option_menu

# import halaman
import Dashboard
import Prediction

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

section[data-testid="stSidebar"]{
    background:#F8FAFC;
    border-right:1px solid #E5E7EB;
}

div[data-testid="stSidebarNav"]{
    display:none;
}

.sidebar-title{
    text-align:center;
    color:#2563EB;
    font-size:40px;
    font-weight:700;
    margin-bottom:10px;
}
            
</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
        Churn Analisis
    </div>
    <hr>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Prediction"
        ],
        icons=[
            "house-door-fill",
            "graph-up-arrow"
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "18px",
                "background-color": "#F8FAFC",
                "border-radius": "12px",
            },
            "icon": {
                "color": "#2563EB",
                "font-size": "22px",
            },
            "nav-link": {
                "font-size": "30px",
                "font-weight": "500",
                "text-align": "left",
                "margin": "8px 0",
                "padding": "12px",
                "border-radius": "10px",
                "--hover-color": "#DBEAFE",
            },
            "nav-link-selected": {
                "background-color": "#2563EB",
                "color": "white",
                "font-weight": "600",
                "border-radius": "10px",
            }
        }
    )

if selected == "Dashboard":
    Dashboard.show()

elif selected == "Prediction":
    Prediction.show()