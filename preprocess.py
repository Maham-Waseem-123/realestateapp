#This import pandas for handing tables(Dataframes)
#and Label encoder for tranding text labels to numbers

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    df['year_sold'] = df['date'].dt.year
    df['month_sold'] = df['date'].dt.month
    df.drop('date', axis=1, inplace=True)

    # Drop ID
    df.drop('id', axis=1, inplace=True)

    # Feature Engineering
    df['house_age'] = 2025 - df['yr_built']
    df['was_renovated'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)
    df['total_sqft'] = df['sqft_living'] + df['sqft_basement']

    # Drop if any missing values exist (optional)
    df.dropna(inplace=True)

    # Return clean dataframe
    return df

