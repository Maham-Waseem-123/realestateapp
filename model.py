from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def train_model(df):
    # Features to use for prediction
    # NEW (5 features)
    X = df[['bedrooms', 'bathrooms', 'total_sqft', 'floors', 'house_age']]

    
    # Target variable (what we want to predict)
    y = df['price']

    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate performance
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model MAE: ${mae:,.2f}")

    # Save the model to use in Streamlit
    joblib.dump(model, 'price_model.pkl')
    return model
