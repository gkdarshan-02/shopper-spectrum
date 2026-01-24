# Libraries used:
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load product recommendation data:
with open("pickle/product_recommends.pkl", "rb") as f:
    dit_products = pickle.load(f)
products = dit_products["products"]
product_vector = dit_products["product_vector"]
product_index = dit_products["index"]

# Recommendation function:
def recommend(desc, n=5):
    if desc not in product_index:
        return None
    id = product_index[desc]
    sim_scores = list(enumerate(cosine_similarity(product_vector[id], product_vector)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top = [i[0] for i in sim_scores[1:n+1]]
    return products.iloc[top].reset_index(drop=True)

# Load customer segmentation model:
with open("pickle/rfm_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
scaler = bundle["scaler"]

# Cluster labels:
rfm_labels = {
    0: "Regular",
    1: "At-Risk",
    2: "High-Value",
    3: "Occasional"
}


# Streamlit UI:

# Page Config:
st.set_page_config(page_title='Shopper Spectrum',layout="centered")
st.title("🛍️ Shopper Spectrum App")

# CSS:
st.markdown("""
    <style>  
    .stAppHeader, [data-testid="stSidebarCollapseButton"] {
        display: none;        
    } 
            
    .stMainBlockContainer {
        padding: 5rem 1rem 1rem;
    }

    p {
        font-size:18px !important;        
    }

    </style>
""", unsafe_allow_html=True)

tab = st.sidebar.radio("Go to", ["📦 Product Recommendations", "👥 Customer Segmentation"])

if tab == "📦 Product Recommendations":
    st.subheader("Find Similar Products")
    inp = st.selectbox("Select a product:", [""] + sorted(products['Description']))
    if st.button("Recommend"):
        if inp == "":
            st.warning("Please select a product.")
        else:
            out = recommend(inp)
            if out is None:
                st.error("Product not found.")
            else:
                st.success("Top Recommendations:")
                for i, row in out.iterrows():
                    st.write(f"• {row['Description']}")

if tab == "👥 Customer Segmentation":
    st.subheader("Predict Customer Segment")
    r = st.number_input("Recency (days)", min_value=0)
    f = st.number_input("Frequency (purchases)", min_value=0)
    m = st.number_input("Monetary (spend)", min_value=0.0)

    if st.button("Predict Cluster"):
        data = np.array([[r, f, m]])
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]
        label = rfm_labels[pred]
        st.success(f"Predicted Segment: **{label}**")