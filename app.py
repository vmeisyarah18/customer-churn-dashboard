import streamlit as st
from streamlit_option_menu import option_menu

# import halaman
import Dashboard
import Prediction

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container{
    padding-top:2rem;
    padding-left:2rem;
    padding-right:2rem;
    max-width:100%;
}
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
div.stButton > button{
    background:#2563EB;
    color:white;
    border-radius:10px;
    border:none;
    font-size:16px;
    font-weight:bold;
}

div.stButton > button:hover{
    background:#1D4ED8;
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

    st.set_page_config
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
            "container":{
                "padding":"8px",
                "background-color":"#FFFFFF",
            },
        
            "icon":{
                "color":"#2563EB",
                "font-size":"20px",
            },
        
            "nav-link":{
                "font-size":"18px",
                "text-align":"left",
                "margin":"5px",
                "color":"#111111",
                "--hover-color":"#EAF2FF",
            },
        
            "nav-link-selected":{
                "background-color":"#2563EB",
                "color":"white",
            },
        }
    )

if selected == "Dashboard":
    Dashboard.show()

elif selected == "Prediction":
    Prediction.show()
