import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer,RobustScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class SinCosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.c_[np.sin(X / self.period * 2 * np.pi), np.cos(X / self.period * 2 * np.pi)]

class DataProcessor:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.features = None

    def load_data(self):
        """Load data from a parquet file."""
        self.data = pd.read_parquet(os.path.join(self.path, 'cleaned_taxi_data.parquet'))
        print(self.data.shape)

    def select_features(self):
        """Select features for analysis."""
        feature_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID', 'total_amount']
        self.features = self.data[feature_cols]

class FeatureEngineering:
    def __init__(self, df):
        self.df = df
        self.hourly_pickups = None
        self.manhattan_df = None

    def generate_hourly_data(self):
        """Aggregate data by hour and handle datetime features."""
        self.hourly_pickups = self.df.groupby(pd.Grouper(freq='h', key='tpep_pickup_datetime')).agg({
            'trip_distance': 'mean',
            'total_amount': 'mean'
        })
        self.hourly_pickups['PU_count'] = self.df.groupby(pd.Grouper(freq='h', key='tpep_pickup_datetime')).size().values
        self.hourly_pickups = self.hourly_pickups.reset_index()
        self.extract_datetime_features()

    def extract_datetime_features(self):
        """Extract datetime and create lag and EWMA features."""
        self.hourly_pickups.insert(loc=2, column='PU_month', value=self.hourly_pickups['tpep_pickup_datetime'].dt.month)
        self.hourly_pickups.insert(loc=3, column='PU_day_of_month', value=self.hourly_pickups['tpep_pickup_datetime'].dt.day)
        self.hourly_pickups.insert(loc=4, column='PU_day_of_week', value=self.hourly_pickups['tpep_pickup_datetime'].dt.weekday)
        self.hourly_pickups.insert(loc=5, column='PU_hour', value=self.hourly_pickups['tpep_pickup_datetime'].dt.hour)
        self.create_lag_features()

    def create_lag_features(self):
        """Add lagged features of the 'PU_count' column."""
        self.hourly_pickups['lag_1h'] = self.hourly_pickups['PU_count'].shift(1).fillna(0)
        self.hourly_pickups['lag_2h'] = self.hourly_pickups['PU_count'].shift(2).fillna(0)
        self.hourly_pickups['lag_1d'] = self.hourly_pickups['PU_count'].shift(24).fillna(0)
        self.hourly_pickups['lag_2d'] = self.hourly_pickups['PU_count'].shift(48).fillna(0)
        self.calculate_ewma()

    def calculate_ewma(self):
        """Calculate the exponentially weighted moving averages."""
        self.hourly_pickups['ewma_3h'] = self.hourly_pickups['PU_count'].ewm(span=3).mean()
        self.hourly_pickups['ewma_6h'] = self.hourly_pickups['PU_count'].ewm(span=6).mean()
        self.hourly_pickups['ewma_12h'] = self.hourly_pickups['PU_count'].ewm(span=12).mean()
        self.hourly_pickups['ewma_24h'] = self.hourly_pickups['PU_count'].ewm(span=24).mean()
        self.calculate_statistics()

    def calculate_statistics(self):
        """Calculate and print statistics."""
        pu_count_range = self.hourly_pickups['PU_count'].max() - self.hourly_pickups['PU_count'].min()
        print(f"Range of PU_count: {pu_count_range}")

        percentiles = self.hourly_pickups['PU_count'].quantile([0.25, 0.5, 0.75])
        print(f"25th Percentile (Q1): {percentiles[0.25]}")
        print(f"50th Percentile (Median): {percentiles[0.5]}")
        print(f"75th Percentile (Q3): {percentiles[0.75]}")

        self.save_data()

    def save_data(self):
        """Save the DataFrame with new features to a CSV file."""
        self.hourly_pickups.to_csv('../Filtered_Dataset/global_temporal_features_withoutPUIDS.csv', index=False)


class ModelTraining:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def prepare_pipeline(self):
        categorical_features = ['PU_hour']
        cyclic_features = {
            'PU_month': 12, 
            'PU_day_of_month': 31, 
            'PU_day_of_week': 7, 
            'PU_hour': 24
        }

        transformers = []
        for feature, period in cyclic_features.items():
            transformers.append((f'{feature}_sin_cos', SinCosTransformer(period), [feature]))

        preprocessor = ColumnTransformer(
            transformers + [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
            remainder='passthrough'
        )

        return preprocessor

    def train_linear_regression(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from joblib import dump

        pipeline = Pipeline([
            ('preprocessor', self.prepare_pipeline()),
            ('regressor', LinearRegression())
        ])
        pipeline.fit(self.X_train, self.y_train)
        self.evaluate_model(pipeline, 'Linear Regression')
        dump(pipeline, 'linear_regression_model.joblib')

    def train_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from joblib import dump

        rf_pipeline = Pipeline([
            ('preprocessor', self.prepare_pipeline()),
            ('regressor', RandomForestRegressor())
        ])
        param_distributions = {
            'regressor__n_estimators': [100, 200, 300, 400],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
        random_search = RandomizedSearchCV(rf_pipeline, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3)
        random_search.fit(self.X_train, self.y_train)
        self.evaluate_model(random_search.best_estimator_, 'Random Forest')
        dump(random_search.best_estimator_, 'random_forest_model.joblib')

    def evaluate_model(self, model, model_name):
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        print(f"{model_name} - Train MAE: {mae_train:.2f}, RMSE: {rmse_train:.2f}")
        print(f"{model_name} - Test MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}")

def main():
    path = "../Filtered_Dataset"
    processor = DataProcessor(path)
    processor.load_data()
    processor.select_features()
    
    fe = FeatureEngineering(processor.features)
    fe.generate_hourly_data()
    
    # Clean and prepare feature data
    global_feat_data = fe.hourly_pickups.copy()
    global_feat_data.drop(['tpep_pickup_datetime', 'total_amount', 'ewma_24h', 'lag_2d'], axis=1, inplace=True)
    global_feat_data.fillna(0, inplace=True)

    # Define explanatory and target features
    X_features = ['PU_month', 'PU_day_of_month', 'PU_day_of_week', 'PU_hour', 'trip_distance', 'lag_1h', 'lag_2h', 'lag_1d', 'ewma_3h', 'ewma_6h', 'ewma_12h']
    y_feature = 'PU_count'

    # Split the data into training and test sets
    train_pct = 0.8
    train_idx = int(train_pct * len(global_feat_data))
    train_data = global_feat_data.iloc[:train_idx]
    test_data = global_feat_data.iloc[train_idx:]

    X_train = train_data[X_features]
    y_train = train_data[y_feature]
    X_test = test_data[X_features]
    y_test = test_data[y_feature]

    # Initialize the model training class and train models
    trainer = ModelTraining(X_train, y_train, X_test, y_test)
    trainer.train_linear_regression()
    trainer.train_random_forest()

if __name__ == '__main__':
    main()