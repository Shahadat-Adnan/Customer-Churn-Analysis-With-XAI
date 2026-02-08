import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Loading Pre Saved Objects
#---------------------------

model = joblib.load("D:\Imp_Files\Codes\Project\Churn_Project/voting_classifier_model.joblib")
encoders = joblib.load("D:\Imp_Files\Codes\Project\Churn_Project/encoders.joblib")
imputers = joblib.load("D:\Imp_Files\Codes\Project\Churn_Project/imputers.joblib")
feature_columns = joblib.load("D:\Imp_Files\Codes\Project\Churn_Project/feature_columns.joblib")
background = pd.read_csv("D:\Imp_Files\Codes\Project\Churn_Project/shap_background.csv")

# Fixing Data Leakage
#----------------------

tgt_col = "churned"


if tgt_col in feature_columns:
    feature_columns = [
        c for c in feature_columns if c != tgt_col
    ]


if tgt_col in imputers:
    imputers.pop(tgt_col)


if tgt_col in encoders:
    encoders.pop(tgt_col)

# Using Gradient Boosting for SHAP
#-----------------------------------
gtb_model = model.named_estimators_["gtb"]

explainer = shap.Explainer(
    gtb_model,
    background,
    model_output="probability"
)

# Feature Mapping
# -----------------

def make_feature_readable(feature_name):
    """
    Converts encoded feature names into readable text.
    """

    if "_" not in feature_name:
        return feature_name.replace("_", " ").title()
    
    else:
        parts = feature_name.split("_", 1)

        base = parts[0].replace("_", " ").title()
        value = parts[1].replace("_", " ").title()

    return f"{base} {value}"

# UI
# -------------

st.set_page_config("Churn Prediction", layout="wide")
st.markdown("""
<style>

/* Remove Streamlit top padding */
.block-container {
    padding-top: 1rem;
}

/* ===== CARD BASE ===== */
.custom-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
}

/* ===== TITLE CARD (SAFE) ===== */
.custom-title {
    border-left: 6px solid #E50914;
    padding-left: 15px;
}

/* ===== METRIC CARD ===== */
.metric-card {
    text-align: center;
    padding: 40px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1c1f26, #111318);
    border: 1px solid #2a2d34;
    margin-bottom: 30px;
}

/* HIGH / LOW CHURN TEXT */
.metric-label {
    font-size: 22px;
    font-weight: 600;
    color: #dddddd;
    margin-bottom: 12px;
}

/* CONFIDENCE PERCENTAGE */
.metric-value {
    font-size: 58px;
    font-weight: 900;
    margin: 10px 0;
}

/* ===== COLORS ===== */
.metric-green {
    color: #2ecc71;
}

.metric-red {
    color: #e74c3c;
}

/* ===== DRIVER BADGES ===== */
.badge {
    display: inline-block;
    background-color: #2a2d34;
    padding: 8px 18px;
    border-radius: 22px;
    margin: 8px 10px 8px 0;
    font-size: 15px;
}
            
/* XAI TEXT */
.xai-text {
    font-size: 18px;
    line-height: 1.7;
    color: #e6e6e6;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Equal top alignment for cards */
.equal-card {
    background-color: #1c1f26;
    padding: 25px;
    border-radius: 16px;
    margin-top: 0px;
}

/* Force identical heading spacing */
.equal-card h3 {
    margin-top: 0px;
    margin-bottom: 18px;
    padding-top: 0px;
}

</style>
""", unsafe_allow_html=True)


st.title("Netflix Customer Churn Prediction")
st.markdown("Machine Learning :  Explainable AI : Real-time Decision Support")
st.divider()
form_left, form_right = st.columns(2)

# Taking User Input
# -------------------
with form_left:
    st.markdown("""<div class="equal-card"><h3>General Information</h3>""", unsafe_allow_html=True)

    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female","Other"])
    device = st.selectbox("Device", ["Mobile", "TV", "Laptop","Tablet", "Desktop"])
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    payment = st.selectbox("Payment Method", ["Gift Card", "Crypto", "Debit Card", "PayPal", "Credit Card"])
    fav_genre = st.selectbox("Favorite Genre", ['Action', 'Sci-Fi', 'Drama', 'Horror', 'Romance', 'Comedy', 'Documentary'])

    st.markdown("</div>", unsafe_allow_html=True)
with form_right:    
    st.markdown("""
    <div class="equal-card"><h3>Usage & Billing</h3>""", unsafe_allow_html=True)

    watch_hours = st.number_input("Total Watch Hours", 0.0)
    avg_watch_time = st.number_input("Avg Watch Time Per Day", 0.0)
    last_login = st.number_input("Days Since Last Login", 0)
    profiles = st.number_input("Number of Profiles", 1, 5)
    monthly_fee = st.number_input("Monthly Fee", 0.0)

    st.markdown("</div>", unsafe_allow_html=True)

# Changing user input into Dataframe
# -------------------------------------

raw_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "subscription_type": subscription,
    "watch_hours": watch_hours,
    "last_login_days": last_login,
    "device": device,
    "monthly_fee": monthly_fee,
    "payment_method": payment,
    "number_of_profiles": profiles,
    "avg_watch_time_per_day": avg_watch_time,
    "favorite_genre":  fav_genre
}])

# Preprocessing Fucntion
# -----------------------

def preprocessing_funct(df):
    df = df.copy()

    # -------- Imputation --------
    for col, imputer in imputers.items():
        if col not in df.columns:
            continue
        else:
            df[col] = imputer.transform(df[[col]]).ravel()

    # -------- Encoding --------
    for col, encoder in encoders.items():
        if col not in df.columns:
            continue
        else:
            encoded = encoder.transform(df[[col]])

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([col]),
                index=df.index
            )

            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)

    # -------- Mapping Features Columns --------
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df

# Prediction & SHAP
# -------------------------
st.divider()
st.markdown("""<div class="equal-card"><h3>Predcition & Explainability</h3>""", unsafe_allow_html=True)
pred_btn = st.button("Predict Churn", use_container_width=True)
if pred_btn:

        X = preprocessing_funct(raw_df)

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        cls_labels = model.classes_

        # Finding index positions
        churn_index = list(cls_labels).index(0)
        not_churn_index = list(cls_labels).index(1)

        if pred == 0:
            confidence = float(proba[churn_index])
            status = "HIGH CHURN RISK"
            color_class = "metric-red"
        else:
            confidence = float(proba[not_churn_index])
            status = "LOW CHURN RISK"
            color_class = "metric-green"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{status}</div>
                <div class="metric-value {color_class}">
                    {confidence:.2%}
                </div>
                <div class="metric-label">Prediction Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# SHAP Dataframe initializatoin
# -------------------------------

        shap_values = explainer(X)

        shap_vals = shap_values.values

        # handles binary & multiclass
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        shap_df = pd.DataFrame({
            "feature": X.columns,
            "shap_value": shap_vals[0]
        })

        shap_df["abs_value"] = shap_df["shap_value"].abs()

        class_labels = model.classes_
        churn_index = list(class_labels).index(0)

        if churn_index == 0:
            shap_df["impact"] = shap_df["shap_value"].apply(
                lambda x: "Which Increases churn risk" if x < 0 else "Which Reduces churn risk"
            )
        else:
            shap_df["impact"] = shap_df["shap_value"].apply(
                lambda x: "Which Increases churn risk" if x > 0 else "Which Reduces churn risk"
            )

        shap_df["feature_readable"] = shap_df["feature"].apply(
            make_feature_readable
        )

        shap_df = shap_df.sort_values("abs_value", ascending=False)

        if pred == 0:
            driver_title = "🔺 Top Churn Drivers"
            driver_df = shap_df[shap_df["impact"].str.contains("Increases")].head(3)
        else:
            driver_title = "🟢 Top Non-Churn Drivers"
            driver_df = shap_df[shap_df["impact"].str.contains("Reduces")].head(3)

        st.markdown(f"### {driver_title}")

        for _, row in driver_df.iterrows():
            st.markdown(
                f"<span class='badge'>{row['feature_readable']}</span>",
                unsafe_allow_html=True
            )


# XAI Explanation Fucntion
# --------------------------

        def gen_explanation(shap_df,prediction,probability,top_k=3):
            """
            Generate a clean, human-readable explanation
            from SHAP values.
            """

            top_features = shap_df.head(top_k)

            if prediction == 1:
                intro = (
                    f"The model predicts that this customer is "
                    f"**unlikely to churn** (with likelyhood: {probability:.2%}). "
                )
            else:
                intro = (
                    f"The model predicts that this customer is "
                    f"**likely to churn** (with likelyhood: {probability:.2%}). "
                )

            reasons = []

            for _, row in top_features.iterrows():
                reasons.append(
                    f"**{row['feature_readable']}** {row['impact'].lower()}"
                )

            explanation = intro + "This decision is mainly influenced by: "

            explanation += "; ".join(reasons) + "."

            return explanation

        st.markdown("### Explainability Table")

        st.dataframe(
            shap_df[
                ["feature_readable", "impact", "abs_value","shap_value"]
            ].head(3)
        )


    # XAI Explnanation
    # ------------------------

        XAI_human_explanation = gen_explanation(
            shap_df=shap_df,
            prediction=pred,
            probability=confidence,
            top_k=3
        )

        st.subheader("XAI Explanation")
        st.markdown(XAI_human_explanation)    
   
    



