# 🛍️ Shopper Spectrum App

An elegant and interactive **Streamlit** web application that delivers real-time **product recommendations** and **customer segmentation** using machine learning. Built from the ground up with deep data cleaning, **EDA**, **vectorization**, **clustering**, and **TF-IDF-based NLP**, this app transforms retail data into actionable insights.

---

## 🌐 Live App

🔗 [Try the app now](https://chz-shopper-spectrum.streamlit.app)

---

## ⚡ Key Features

- 🛒 **Product Recommendations** based on item similarity using TF-IDF & cosine similarity
- 🧠 **Customer Segmentation** using KMeans clustering with RFM modeling
- 🔄 Supports both NLP-based and behavioral segmentation in one app
- 🔍 Choose any product and discover the top 5 most similar items
- 📊 Enter customer RFM metrics to predict their cluster
- 💡 Interpretable cluster labels like _High-Value_, _Regular_, _At-Risk_, and more
- 💻 Responsive and clean Streamlit interface with internal CSS styling

---

## 🌟 What Makes It Special?

**From Purchase Behavior to Predictions.**  
This app blends NLP with unsupervised learning to power a dynamic customer intelligence system. Whether you're a data analyst, business strategist, or a curious shopper — it empowers you to understand user behavior and improve marketing targeting or inventory planning.

---

## 🛠️ Tech Stack

| Tool           | Purpose                      |
| -------------- | ---------------------------- |
| Streamlit      | Web UI framework             |
| Pandas / NumPy | Data manipulation            |
| Sk-learn       | ML models (KMeans, TF-IDF)   |
| Matplotlib     | Visualizations               |
| Pickle         | Model & object serialization |
| CSS (inline)   | UI theming and styling       |

---

## 🖼️ Screenshots

![Recommendation Page](screenshots/Product_Recommends.png)  
![Segmentation Page](screenshots/Customer_Segment.png)

---

## 🧠 Clustering Logic

| Segment Label | Characteristics                                   |
| ------------- | ------------------------------------------------- |
| High-Value    | High R, High F, High M – frequent, big spenders   |
| Regular       | Medium F, Medium M – steady buyers                |
| Occasional    | Low F, Low M, older R – rare, inconsistent        |
| At-Risk       | High R, Low F, Low M – haven't purchased recently |

---
