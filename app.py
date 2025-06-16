import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model = joblib.load('price_model.pkl')
df = pd.read_csv('property_data.csv')

# App layout
st.title("🏠 Real Estate Price Estimator")
st.subheader("Predict house price using key features")

# Sidebar inputs
with st.sidebar:
    st.header("Enter Property Details")
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    total_sqft = st.number_input("Total Square Foot Area", 500, 10000, 1500)
    floors = st.slider("Floors", 1, 3, 1)
    house_age = st.slider("House Age", 0, 120, 30)

# Predict price
if st.button("Estimate Price"):
    input_data = [[bedrooms, bathrooms, total_sqft, floors, house_age]]
    prediction = model.predict(input_data)[0]
    st.success(f"🏷️ Estimated House Price: **${prediction:,.2f}**")

# Data Visualization
st.divider()
st.subheader("📈 Market Trends")

st.markdown("""
Understanding market trends helps you make smarter real estate decisions.

These charts below show how different features like bedrooms, house size, and time affect house prices. 
You can use this insight to:
- Estimate fair property prices
- Understand price growth over time
- Compare properties based on their features

Let's explore the market visually:
""")


# 🔹 2. Average Price by Number of Bedrooms
st.subheader("💤 Average Price by Number of Bedrooms")
bed_price = df.groupby('bedrooms')['price'].mean().reset_index()
fig1, ax1 = plt.subplots()
sns.barplot(data=bed_price, x='bedrooms', y='price', ax=ax1)
ax1.set_title("Average Price vs Bedrooms")
st.pyplot(fig1)

# 🔹 3. Price vs. Total Square Foot Area
st.subheader("📏 Price vs. Total Square Foot Area")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='total_sqft', y='price', ax=ax2)
ax2.set_title("Price vs. Total Sqft")
st.pyplot(fig2)

# 🔹 4. Price Trend by Year Sold
st.subheader("📅 Price Trend by Year Sold")
yearly_price = df.groupby('year_sold')['price'].mean().reset_index()
fig3, ax3 = plt.subplots()
sns.lineplot(data=yearly_price, x='year_sold', y='price', marker='o', ax=ax3)
ax3.set_title("Average Price by Year")
st.pyplot(fig3)
