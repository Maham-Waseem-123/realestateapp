# 🏠 Real Estate Price Estimator

This Streamlit web app predicts house prices based on user inputs like number of bedrooms, bathrooms, total square footage, floors, and house age. It also visualizes market trends based on real estate data.

## 🚀 Features

- 🔢 Predict house price using a trained Random Forest model
- 📊 View market trends:
  - Price vs. bedrooms
  - Price vs. house size
  - Price trends over years
- 🎨 Easy-to-use interactive interface built with Streamlit
- 📁 Built with Python, Pandas, Scikit-learn, Matplotlib, and Seaborn

---

## 🧠 How It Works

1. Preprocessed dataset using `preprocess.py`
2. Trained a machine learning model (`RandomForestRegressor`) on 5 features:
   - `bedrooms`, `bathrooms`, `total_sqft`, `floors`, `house_age`
3. Saved the trained model as `price_model.pkl`
4. Used Streamlit in `app.py` to:
   - Accept user input
   - Predict price using the model
   - Show market trend charts

---

## 💻 Run Locally

1. Clone this repository:
```bash
git clone https://github.com/your-username/real-estate-app.git
cd real-estate-app
