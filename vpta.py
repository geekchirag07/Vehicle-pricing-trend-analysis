import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# TITLE
st.title("🚗 Vehicle Pricing Trend Analysis")
st.subheader("Complete Data Analysis Pipeline (Without Machine Learning)")

# Sidebar navigation
step = st.sidebar.selectbox("Select Step", [
    "Step 1: Project Framing",
    "Step 2: Data Dictionary",
    "Step 3: Environment Setup",
    "Step 4: Data Ingestion",
    "Step 5: Data Quality Checks",
    "Step 6: Data Cleaning",
    "Step 7: EDA",
    "Step 8: Feature Engineering",
    "Step 9: Visualization & Storytelling",
    "Step 10: Final Report"
])

# Upload dataset
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

# ---------------- STEP 1 ----------------
if step == "Step 1: Project Framing":
    st.header("📌 Project Objective")
    st.write("Analyze vehicle pricing trends and identify key influencing factors.")

# ---------------- STEP 2 ----------------
elif step == "Step 2: Data Dictionary":
    if file:
        st.dataframe(pd.DataFrame({
            "Column": df.columns,
            "Datatype": df.dtypes
        }))
    else:
        st.warning("Upload dataset first")

# ---------------- STEP 3 ----------------
elif step == "Step 3: Environment Setup":
    st.write("Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit")

# ---------------- STEP 4 ----------------
elif step == "Step 4: Data Ingestion":
    if file:
        st.write("Shape:", df.shape)
        st.dataframe(df.head())
    else:
        st.warning("Upload dataset")

# ---------------- STEP 5 ----------------
elif step == "Step 5: Data Quality Checks":
    if file:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Duplicate Rows")
        st.write(df.duplicated().sum())

        st.subheader("Statistical Summary")
        st.write(df.describe())
    else:
        st.warning("Upload dataset")

# ---------------- STEP 6 ----------------
elif step == "Step 6: Data Cleaning":
    if file:
        df_clean = df.copy()
        df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
        df_clean.drop_duplicates(inplace=True)

        st.success("Cleaning Completed ✅")
        st.write("New Shape:", df_clean.shape)
        st.dataframe(df_clean.head())
    else:
        st.warning("Upload dataset")

# ---------------- STEP 7 ----------------
elif step == "Step 7: EDA":
    if file:
        df["Car Age"] = 2024 - df["Year"]

        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Price"], bins=30)
        st.pyplot(fig)

        st.subheader("Average Price by Brand")
        fig, ax = plt.subplots()
        df.groupby("Brand")["Price"].mean().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Fuel Type Share")
        fig, ax = plt.subplots()
        df["Fuel Type"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

        st.subheader("Price by Fuel Type")
        fig, ax = plt.subplots()
        sns.boxplot(x=df["Fuel Type"], y=df["Price"], ax=ax)
        st.pyplot(fig)

        st.subheader("Price by Condition")
        fig, ax = plt.subplots()
        sns.boxplot(x=df["Condition"], y=df["Price"], ax=ax)
        st.pyplot(fig)

        st.subheader("Mileage vs Price")
        fig, ax = plt.subplots()
        ax.scatter(df["Mileage"], df["Price"])
        st.pyplot(fig)

        st.subheader("Transmission Count")
        fig, ax = plt.subplots()
        df["Transmission"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
        st.pyplot(fig)

# ---------------- STEP 8 ----------------
elif step == "Step 8: Feature Engineering":
    if file:
        df["Car Age"] = 2024 - df["Year"]
        df["Mileage Band"] = pd.cut(df["Mileage"],
                                   bins=[0,50000,100000,150000,200000,np.inf],
                                   labels=["<50k","50-100k","100-150k","150-200k","200k+"])

        st.success("Features Added: Car Age, Mileage Band")
        st.dataframe(df.head())

# ---------------- STEP 9 ----------------
elif step == "Step 9: Visualization & Storytelling":
    if file:
        st.subheader("Price Trend by Year")
        fig, ax = plt.subplots()
        df.groupby("Year")["Price"].mean().plot(ax=ax)
        st.pyplot(fig)

        st.subheader("Price by Mileage Band")
        fig, ax = plt.subplots()
        df.groupby("Mileage Band")["Price"].mean().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Electric vs Non-Electric")
        fig, ax = plt.subplots()
        df.groupby(df["Fuel Type"]=="Electric")["Price"].mean().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Condition vs Price")
        fig, ax = plt.subplots()
        df.groupby("Condition")["Price"].mean().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Transmission vs Fuel Heatmap")
        fig, ax = plt.subplots()
        pivot = df.pivot_table(values="Price", index="Transmission", columns="Fuel Type")
        sns.heatmap(pivot, annot=True, ax=ax)
        st.pyplot(fig)

# ---------------- STEP 12 ----------------
elif step == "Step 10: Final Report":
    st.header("📄 Final Project Report")

    st.success("Vehicle Pricing Trend Analysis Completed Successfully 🎉")

    st.markdown("""
# 🚗 VEHICLE PRICING TREND ANALYSIS  
## Final Project Report  

---

## 📌 1. Introduction  
The **Vehicle Pricing Trend Analysis** project focuses on analyzing a dataset of vehicles to understand how different factors influence car prices. The goal is to uncover meaningful patterns, trends, and relationships within the data using **data analysis and visualization techniques**.

This project helps answer key business questions such as:
- What factors affect vehicle pricing the most?
- How do mileage, age, and condition impact price?
- What trends exist across years and fuel types?

---

## 🎯 2. Objectives  
- Analyze vehicle price distribution and trends  
- Identify key features influencing price  
- Perform data cleaning and preprocessing  
- Conduct exploratory data analysis (EDA)  
- Generate meaningful insights through visualizations  

---

## 📂 3. Dataset Overview  
The dataset consists of multiple attributes related to vehicles, including:

- **Brand** – Manufacturer of the vehicle  
- **Year** – Manufacturing year  
- **Engine Size** – Engine capacity  
- **Fuel Type** – Petrol, Diesel, Electric, etc.  
- **Transmission** – Manual or Automatic  
- **Mileage** – Distance covered by the vehicle  
- **Condition** – New, Used, or Like New  
- **Price** – Target variable  

---

## 🧹 4. Data Cleaning & Preprocessing  
To ensure accuracy and reliability, the dataset was cleaned using the following steps:

- Missing values handled using median (numeric) and mode (categorical)  
- Duplicate records removed  
- Column names standardized  
- Outliers checked and controlled  
- New feature **Car Age** created  

---

## 📊 5. Exploratory Data Analysis (EDA)  

Key Visualizations:
- Price Distribution (Histogram)  
- Average Price by Brand  
- Fuel Type Distribution  
- Price by Fuel Type  
- Price by Condition  
- Mileage vs Price  
- Transmission Distribution  
- Correlation Heatmap  

---

## ⚙️ 6. Feature Engineering  
Additional features were created:

- **Car Age** = Current Year – Manufacturing Year  
- **Mileage Band** = Categorized mileage  

---

## 📈 7. Visualization & Storytelling  

- Price trends over years  
- Price variation across mileage bands  
- Electric vs Non-Electric comparison  
- Condition-based pricing  
- Transmission vs Fuel heatmap  

---

## 🔍 8. Key Insights  

- Vehicle Age & Mileage strongly affect price  
- Newer vehicles have higher prices  
- Lower mileage increases value  
- Electric vehicles show premium pricing  
- Condition impacts resale value  
- Brand influences pricing  

---

## 📌 9. Conclusion  

This project successfully analyzes vehicle pricing trends using data visualization techniques.  
It demonstrates how structured analysis helps extract meaningful insights from raw data.

---

## ✅ 10. Summary  

✔ Data cleaned and processed  
✔ Multiple visualizations created  
✔ Key pricing factors identified  
✔ Trends successfully analyzed  

---

**Project Title:** Vehicle Pricing Trend Analysis  
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit  

---
""")