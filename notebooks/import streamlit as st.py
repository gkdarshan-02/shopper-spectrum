# ============================================================
# Shopper Spectrum – Customer Segmentation & Product Recommendation
# Streamlit Application (app.py)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🛒 Shopper Spectrum – Customer Segmentation & Product Recommendation",
    layout="wide"
)

# ============================================================
# LOAD SAVED MODELS (NO RETRAINING)
# ============================================================
@st.cache_resource
def load_models():
    kmeans_model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    product_similarity = joblib.load("product_similarity.pkl")
    return kmeans_model, scaler, product_similarity

kmeans_model, scaler, product_similarity = load_models()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📌 Navigation")
module = st.sidebar.radio(
    "Select Module",
    ["Product Recommendation", "Customer Segmentation"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Shopper Spectrum\n\n"
    "Customer Segmentation & Product Recommendation\n"
    "E-Commerce Analytics Project"
)

# ============================================================
# MAIN TITLE
# ============================================================
st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation")
st.divider()

# ============================================================
# MODULE 1: PRODUCT RECOMMENDATION
# ============================================================
if module == "Product Recommendation":

    st.header("🎯 Product Recommendation")
    st.write("Recommend similar products using item-based collaborative filtering.")
    st.divider()

    product_name = st.text_input(
        "Enter Product Name",
        placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER"
    )

    if st.button("Get Recommendations"):
        if product_name.strip() == "":
            st.warning("Please enter a product name.")
        elif product_name not in product_similarity.index:
            st.warning("Product not found. Please check the name and try again.")
        else:
            # Fetch similarity scores
            similarity_scores = (
                product_similarity[product_name]
                .sort_values(ascending=False)
            )

            # Exclude the queried product itself
            recommendations = similarity_scores.iloc[1:6]

            st.success("Top 5 Similar Products:")
            for i, prod in enumerate(recommendations.index, start=1):
                st.markdown(f"**{i}. {prod}**")

            st.info(
                "These products are frequently purchased together or by similar customers."
            )

# ============================================================
# MODULE 2: CUSTOMER SEGMENTATION
# ============================================================
elif module == "Customer Segmentation":

    st.header("🎯 Customer Segmentation")
    st.write("Predict customer segment based on RFM inputs.")
    st.divider()

    # -------------------------
    # USER INPUTS
    # -------------------------
    recency = st.number_input(
        "Recency (Days since last purchase)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency (Number of purchases)",
        min_value=0,
        value=5
    )

    monetary = st.number_input(
        "Monetary (Total spend)",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Customer Segment"):

        # Prepare input for model
        input_data = np.array([[recency, frequency, monetary]])

        # Scale input using saved scaler
        input_scaled = scaler.transform(input_data)

        # Predict cluster
        cluster = int(kmeans_model.predict(input_scaled)[0])

        # ------------------------------------------------
        # CLUSTER TO SEGMENT MAPPING
        # NOTE:
        # Mapping is based on RFM behavior interpretation
        # ------------------------------------------------
        segment_mapping = {
            0: ("High-Value", "Highly engaged customers with recent, frequent, and high-value purchases."),
            1: ("Regular", "Consistent customers with moderate purchase frequency and spending."),
            2: ("Occasional", "Infrequent customers with low spending and irregular purchases."),
            3: ("At-Risk", "Customers who have not purchased recently and show declining engagement.")
        }

        segment_name, segment_desc = segment_mapping.get(
            cluster,
            ("Unknown", "Segment definition not available.")
        )

        st.success(f"Predicted Segment: **{segment_name}**")
        st.info(segment_desc)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "© Shopper Spectrum | Customer Segmentation & Recommendation System | Streamlit Deployment"
)

