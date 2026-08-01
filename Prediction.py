import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px

def show():

    st.title("Customer Churn Prediction")
    st.subheader("Masukkan data pelanggan untuk memprediksi kemungkinan churn")

    # =========================
    # LOAD MODEL
    # =========================
    model = joblib.load("Model.pkl")

    print(model.get_booster().feature_names)


    # ==========================================
    # ROW 1
    # ==========================================
    col_input, col_result = st.columns([1, 1.5])

    # ==========================================
    # INPUT DATA
    # ==========================================
    with col_input:

        st.subheader("1. Input Data Pelanggan")

        a, b, c = st.columns(3)

        tenure = a.number_input(
        "Tenure (bulan)",
        min_value=0,
        max_value=72,
        key="tenure"
        )

        contract = b.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            key="contract"
        )

        internet = c.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            key="internet"
        )

        monthly = a.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=150.0,
            key="monthly"
        )

        total = b.number_input(
            "Total Charges (Rp)",
            min_value=0.0,
            max_value=10000.0,
            key="total"
        )

        payment = c.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            key="payment"
        )

        paperless = a.selectbox(
            "Paperless Billing",
            ["No", "Yes"],
            key="paperless"
        )

        online_security = b.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            key="online_security"
        )

        tech = c.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
            key="tech"
        )

        movie = a.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
            key="movie"
        )

        tv = b.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            key="tv"
        )

        device = c.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
            key="device"
        )

        multiple = a.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
            key="multiple"
        )

        online_backup = b.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
            key="online_backup"
        )

        senior = c.selectbox(
            "Senior Citizen",
            [0, 1],
            key="senior"
        )

        partner = a.selectbox(
            "Partner",
            ["No", "Yes"],
            key="partner"
        )

        dependents = b.selectbox(
            "Dependents",
            ["No", "Yes"],
            key="dependents"
        )
        gender = a.selectbox(
            "Gender",
            ["Female", "Male"],
            key="gender"
        )
        phone = b.selectbox(
            "Phone Service",
            ["No", "Yes"],
            key="phone"
        )

        predict_btn = st.button(
            "✨ Prediksi Churn",
            use_container_width=True,
            type="primary"
        )

    # ==========================================
    # PROSES PREDIKSI
    # ==========================================
    if predict_btn:

        input_dict = {
        "SeniorCitizen": senior,
        "gender": gender,
        "tenure": tenure,
        "PhoneService": phone,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Partner": partner,
        "Dependents": dependents,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movie,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment
    }

        input_df = pd.DataFrame([input_dict])

        input_df = pd.get_dummies(input_df)
        
        model_features = model.get_booster().feature_names
        
        input_df = input_df.reindex(
            columns=model_features,
            fill_value=0
        )

        # Ubah semua kolom menjadi integer
        input_df = input_df.astype("int64")
        print(input_df)
        print(input_df.dtypes)
        prob = model.predict_proba(input_df)[0][1]

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Impact": shap_values[0]
        })

        # pastikan semua value numerik
        shap_df["Value"] = input_df.iloc[0].astype(float).values

        # nilai absolut SHAP
        shap_df["abs"] = shap_df["Impact"].abs()

        # urutkan berdasarkan pengaruh terbesar
        shap_sorted = shap_df.sort_values(
            by="abs",
            ascending=False
        )

        # hanya fitur yang aktif
        active_features = shap_df[
            (shap_df["Value"] == 1)
            |
            (~shap_df["Feature"].str.contains("_"))
        ]

        # ambil 10 paling berpengaruh
        top10 = (
            active_features
            .sort_values("abs", ascending=False)
            .head(10)
        )

        positive = (
            top10[top10["Impact"] > 0]
            .sort_values("Impact", ascending=False)
        )

        negative = (
            top10[top10["Impact"] < 0]
            .sort_values("Impact")
        )

        # faktor paling berpengaruh
        top_feature = shap_sorted.iloc[0]

        print(shap_df["Value"].apply(type).value_counts())
        # ======================================
        # HASIL PREDIKSI
        # ======================================
        with col_result:

            st.subheader("2. Hasil Prediksi")

            left_card, right_card = st.columns(2)

            with left_card:

                if prob >= 0.45:
                    label = "⚠️ CHURN"
                    bg_color = "#fde8e8"
                    text_color = "#dc2626"
                else:
                    label = "✅ TIDAK CHURN"
                    bg_color = "#e8f5e9"
                    text_color = "#16a34a"

                st.markdown(
                    f"""
                    <div style="
                        background-color:{bg_color};
                        padding:35px;
                        border-radius:10px;
                        height:160px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:36px;
                        font-weight:bold;
                        color:{text_color};
                    ">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if prob >= 0.70:
                st.error("🚨 High Risk Customer - fokus retensi!")

            elif prob >= 0.45:
                st.warning("⚠️ Medium Risk Customer - perlu monitoring")

            else:
                st.success("✅ Low Risk Customer - pelanggan relatif stabil")

            with right_card:

                st.markdown(
                    "<center><h4>Probabilitas Churn</h4></center>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <center>
                    <h1 style='color:red'>
                    {prob*100:.1f}%
                    </h1>
                    </center>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(float(prob))

            m1, m2, m3, m4, m5 = st.columns([1,1.5,1,1,1])

            m1.metric("Tenure", f"{tenure} bulan")
            m2.metric("Contract", contract)
            m3.metric("Internet", internet)
            m4.metric("Monthly", f"Rp {monthly:,.0f}")
            m5.metric("Total", f"Rp {total:,.0f}")

        # ======================================
        # ======================================
        # FUNCTION RENAME FEATURE
        # ======================================
        def rename_feature(feature):

            if feature == "tenure":
                return f"Tenure ({tenure} bulan)"

            if feature == "MonthlyCharges":
                return f"Monthly Charges (Rp {monthly:,.0f})"

            if feature == "TotalCharges":
                return f"Total Charges (Rp {total:,.0f})"

            replacements = {
                "Contract_": "Contract = ",
                "InternetService_": "Internet Service = ",
                "PaymentMethod_": "Payment Method = ",
                "OnlineSecurity_": "Online Security = ",
                "OnlineBackup_": "Online Backup = ",
                "TechSupport_": "Tech Support = ",
                "DeviceProtection_": "Device Protection = ",
                "StreamingTV_": "Streaming TV = ",
                "StreamingMovies_": "Streaming Movies = ",
                "PaperlessBilling_": "Paperless Billing = ",
                "MultipleLines_": "Multiple Lines = ",
                "Partner_": "Partner = ",
                "Dependents_": "Dependents = ",
                "PhoneService": "Phone"
            }

            for old, new in replacements.items():
                if feature.startswith(old):
                    return feature.replace(old, new)

            return feature

        # ======================================
        # SHAP & REKOMENDASI
        # ======================================
        row3_left, row3_right = st.columns(2)

        with row3_left:

            st.subheader("3. Top Faktor Penyebab Churn")

            st.caption(
            "Faktor yang paling mempengaruhi hasil prediksi berdasarkan nilai SHAP."
        )

            col_pos, col_neg = st.columns(2)

            # =========================
            # MENINGKATKAN RISIKO
            # =========================
            with col_pos:

                st.markdown(
                    "<h5 style='color:#dc2626'>🔴 Meningkatkan Risiko Churn</h5>",
                    unsafe_allow_html=True
                )

                for _, row in positive.iterrows():

                    feature = rename_feature(row["Feature"])
                    impact = abs(float(row["Impact"]))

                    st.markdown(
                        f"""
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            margin-top:8px;
                        ">
                            <span>{feature}</span>
                            <span style="color:#dc2626">
                                +{row['Impact']:.2f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(min(impact, 1.0))

            # =========================
            # MENURUNKAN RISIKO
            # =========================
            with col_neg:

                st.markdown(
                    "<h5 style='color:#16a34a'>🟢 Menurunkan Risiko Churn</h5>",
                    unsafe_allow_html=True
                )

                for _, row in negative.iterrows():

                    feature = rename_feature(row["Feature"])
                    impact = abs(float(row["Impact"]))

                    st.markdown(
                        f"""
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            margin-top:8px;
                        ">
                            <span>{feature}</span>
                            <span style="color:#16a34a">
                                {row['Impact']:.2f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(min(impact, 1.0))

        # ======================================
        # REKOMENDASI
        # ======================================
        with row3_right:

            st.subheader("4. Rekomendasi & Saran")

            recommendations = []

            for _, row in positive.iterrows():

                feature = row["Feature"]

                if "InternetService_Fiber optic" in feature:
                    recommendations.append(
                        "Evaluasi kualitas layanan Fiber Optic, lakukan monitoring gangguan dan lakukan follow-up kepuasan pelanggan."
                    )
                elif "InternetService_No" in feature:
                    recommendations.append(
                        "Analisis kebutuhan pelanggan dan tawarkan layanan internet apabila sesuai dengan profil dan kebutuhan mereka."
                    )
                elif "PaymentMethod_Electronic check" in feature:
                    recommendations.append(
                        "Dorong pelanggan beralih ke pembayaran otomatis (AutoPay) melalui kartu kredit atau transfer bank dengan insentif tertentu."
                    )

                elif "tenure" == feature:
                    recommendations.append(
                        "Berikan program loyalitas atau penawaran khusus untuk pelanggan baru."
                    )

                elif "MonthlyCharges" == feature:
                    recommendations.append(
                        "Tawarkan paket layanan yang lebih sesuai dengan kebutuhan pelanggan, diskon atau bundling layanan agar biaya terasa lebih bernilai."
                    )

                elif "PaperlessBilling_Yes" in feature:
                    recommendations.append(
                        "Kirim pengingat tagihan dan informasi pembayaran secara berkala melalui, email,SMS atau notifikasi aplikasi."
                    )

                elif "TechSupport_No" in feature:
                    recommendations.append(
                        "Tingkatkan penggunaan layanan Tech Support untuk membantu pelanggan dan tawarkan paket yang sudah termasuk layanan tersebut."
                    )

                elif "OnlineSecurity_No" in feature:
                    recommendations.append(
                        "Tawarkan fitur keamanan tambahan untuk meningkatkan kepuasan pelanggan."
                    )
                elif "OnlineBackup_No" in feature:
                    recommendations.append(
                        "Promosikan layanan pencadangan data (Online Backup) melalui paket bundling atau harga promosi."
                    )
                elif "Contract_Month-to-month" in feature:
                    recommendations.append(
                        "Tawarkan kontrak jangka panjang dengan potongan harga atau bonus layanan."
                    )
                elif "StreamingMovies_Yes / StreamingTV_Yes" in feature:
                    recommendations.append(
                        "Pastikan kualitas jaringan tetap stabil dan rekomendasikan paket internet yang sesuai untuk aktivitas streaming."
                    )
                elif "Contract_One year" in feature:
                    recommendations.append(
                        "Dorong pelanggan untuk memperpanjang kontrak dengan memberikan penawaran yang menarik sebelum masa kontrak berakhir."
                    )
                elif "Contract_Two year" in feature:
                    recommendations.append(
                        "Pertahankan pelanggan dengan menjaga kualitas layanan dan berikan penghargaan loyalitas menjelang berakhirnya kontrak."
                    )
                elif "DeviceProtection_No" in feature:
                    recommendations.append(
                        "Tawarkan layanan perlindungan perangkat untuk meningkatkan nilai layanan yang diterima pelanggan."
                    )

            recommendations = list(set(recommendations))

            st.info("💡 Rekomendasi Utama")

            if recommendations:
                for r in recommendations:
                    st.write("✔️", r)
            else:
                st.write(
                    "✔️ Pelanggan tidak menunjukkan faktor risiko churn yang signifikan."
                )


