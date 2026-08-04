import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import shap
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)

def show():

    st.markdown("""
    <h1 style='color:#2563EB'>
    📈 Customer Churn Dashboard
    </h1>
    """, unsafe_allow_html=True)
    st.subheader("Ringkasan statistik, performa model, dan hasil analisis customer churn pada dataset Telco Customer Churn")

    model = joblib.load("Model.pkl")
    x_test = pd.read_csv("data/x_test.csv")
    y_test = pd.read_csv("data/y_test.csv")
    y_test = y_test.values.ravel()
    y_prob = model.predict_proba(x_test)[:,1]

    best_threshold = 0.45
    y_pred = (y_prob >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # =========================
    # SHAP FEATURE IMPORTANCE
    # =========================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    shap_importance = pd.DataFrame({
        "Feature": x_test.columns,
        "Importance": np.abs(shap_values).mean(axis=0)
    })

    top5_segment = (
        shap_importance
        .sort_values("Importance", ascending=False)
        .head(5)
    )
    top5_segment["Importance"] = top5_segment["Importance"].astype(float)

    st.markdown("""
    <style>

    /* ======================
    BACKGROUND
    ====================== */
    .stApp{
        background-color: white;
        font-family: "Segoe UI", sans-serif;
    }

    /* ======================
    METRIC CARD
    ====================== */
    div[data-testid="stMetric"]{
        background:white;
        border:1px solid #e5e7eb;
        padding:15px;
        border-radius:15px;
        box-shadow:0 2px 10px rgba(0,0,0,.08);
    }

    /* Judul metric */
    div[data-testid="stMetricLabel"]{
        font-size:18px !important;
        font-weight:600;
        color:#374151;
    }

    /* Angka metric */
    div[data-testid="stMetricValue"]{
        font-size:36px !important;
        font-weight:700;
        color:#1e293b;
    }

    /* ======================
    CARD / CONTAINER
    ====================== */
    div[data-testid="stVerticalBlockBorderWrapper"]{
        background:white;
        border-radius:15px;
        padding:18px;
        box-shadow:0 2px 10px rgba(0,0,0,.08);
    }
                

    </style>
    """, unsafe_allow_html=True)


    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


    # =========================
    # KPI
    # =========================
    total_customer = len(df)
    churn_yes = (df["Churn"] == 1).sum()
    churn_rate = churn_yes / total_customer * 100
    total_revenue = df["TotalCharges"].sum()
    high_risk = (y_prob >= best_threshold).sum()

    # =========================
    # STATISTIK
    # =========================
    churn_counts = df["Churn"].value_counts()
    avg_monthly = df["MonthlyCharges"].mean()
    avg_total = df["TotalCharges"].mean()

    # =========================
    # ROW 1
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("👥 Total Customer", f"{total_customer:,}")

    with c2:
        st.metric("📉 Churn Rate", f"{churn_rate:.2f}%")

    with c3:
        st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")

    with c4:
        st.metric("⚠️ High Risk Customer", f"{high_risk:,}")

    st.markdown("""
    <style>
    
    /* Dashboard */
    .block-container{
        padding-top:1rem;
        padding-bottom:1rem;
    }
    
    /* Card */
    .card{
        background:white;
        border:1px solid #E5E7EB;
        border-radius:14px;
        padding:18px;
        box-shadow:0 2px 8px rgba(0,0,0,.05);
        margin-bottom:12px;
    }
    
    /* Judul kecil */
    .card-title{
        font-size:16px;
        color:#6b7280;
        font-weight:600;
    }
    
    /* Angka besar */
    .card-value{
        font-size:34px;
        font-weight:700;
        color:#1f2937;
    }
    
    /* Informasi model */
    .info-row{
        padding:12px 0;
        border-bottom:1px solid #ECECEC;
    }
    
    /* Metric */
    div[data-testid="stMetric"]{
        background:white;
        border:1px solid #E5E7EB;
        border-radius:12px;
        padding:14px;
        box-shadow:0 2px 6px rgba(0,0,0,.05);
    }
    
    </style>
    """, unsafe_allow_html=True)
    # =========================
    # ROW 1
    # =========================
    col1,col2,col3 = st.columns([1.15,1.15,1.7], gap="medium")

    # =========================
    # DISTRIBUSI CHURN
    # =========================
    with col1:
        with st.container(border=True):

            st.subheader("Distribusi Churn (Yes vs No)")

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Churn = No","Churn = Yes"],
                        values=[churn_counts[0], churn_counts[1]],
                        marker=dict(colors=["#2563eb","#ef4444"]),
                        hole=0,
                        textinfo="percent",
                        textfont_size=20,
                        sort=False
                    )
                ]
            )

            fig.update_layout(
                template="plotly_white",
                height=480,
                showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10)
            )

            st.plotly_chart(fig, use_container_width=True)
    # =========================
    # RATA-RATA CHARGES
    # =========================
    with col2:

        with st.container(border=True):

            st.subheader("Rata-rata Charges")

            avg_tenure = df["tenure"].mean()

            st.markdown(f"""
            <div class="card">
            
            <div class="card-title">
            Monthly Charges
            </div>
            
            <div class="card-value">
            ${avg_monthly:.2f}
            </div>
            
            <hr>
            
            <div class="card-title">
            Total Charges
            </div>
            
            <div class="card-value">
            ${avg_total:.2f}
            </div>
            
            <hr>
            
            <div class="card-title">
            Average Tenure
            </div>
            
            <div class="card-value">
            {avg_tenure:.1f} Months
            </div>
            
            </div>
            """, unsafe_allow_html=True)
    # =========================
    # INFORMASI MODEL
    # =========================
    with col3:
        with st.container(border=True):

            st.subheader("Informasi Model")

            st.markdown(f"""
            <div class="card">
            
            <div class="info-row">
            📋 <b>Algoritma</b><br>
            XGBoost Classifier
            </div>
            
            <div class="info-row">
            🎯 <b>Target</b><br>
            Customer Churn
            </div>
            
            <div class="info-row">
            ⚙️ <b>Metode Validasi</b><br>
            Train-Test Split (80 : 20)
            
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            st.subheader("Performa Model (Data Test)")

            r1,r2,r3 = st.columns(3)

            with r1:
                st.metric("Accuracy",f"{accuracy:.3f}")

            with r2:
                st.metric("Precision",f"{precision:.3f}")

            with r3:
                st.metric("Recall",f"{recall:.3f}")
            
            r4, r5 = st.columns(2)
            with r4:
                st.metric("F1-Score",f"{f1:.3f}")

            with r5:
                st.metric("ROC-AUC",f"{roc_auc:.3f}")

    # =========================
    # TOP CUSTOMER + TOP SHAP
    # =========================
    st.divider()

    col_left, col_right = st.columns([1,1.2])

    with col_left:

        with st.container(border=True):

            st.subheader("Daftar Pelanggan dengan Probabilitas Churn Tertinggi")

            df_customer = x_test.copy()

            df_customer["customerID"] = df["customerID"].iloc[:len(x_test)].values

            df_customer["Churn Probability (%)"] = (y_prob * 100).round(2)

            df_customer["Prediksi"] = np.where(
                y_prob >= best_threshold,
                "Churn",
                "Not Churn"
            )

            top10 = (
                df_customer
                .sort_values(
                    "Churn Probability (%)",
                    ascending=False
                )
                .head(10)
            )

            top10_show = top10[
                [
                    "customerID",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "Churn Probability (%)",
                    "Prediksi"
                ]
            ].copy()

            top10_show.columns = [
                "Customer ID",
                "Tenure",
                "Monthly Charges",
                "Total Charges",
                "Churn Probability (%)",
                "Prediction"
            ]

            st.dataframe(
                top10_show,
                use_container_width=True,
                hide_index=True
            )


    with col_right:
        with st.container(border=True):

            st.subheader("Top 5 Faktor Churn (SHAP)")

            fig_shap = go.Figure()

            fig_shap.add_trace(
                go.Bar(
                    x=top5_segment["Importance"],
                    y=top5_segment["Feature"],
                    orientation="h",
                    text=top5_segment["Importance"].round(3),
                    textposition="outside"
                )
            )

            fig_shap.update_layout(
                template="plotly_white",
                height=340,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis_title="SHAP Importance",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                showlegend=False
            )

            st.plotly_chart(
                fig_shap,
                use_container_width=True
            )

            st.info(
                "Faktor di atas diperoleh dari analisis SHAP pada model XGBoost."
            )

