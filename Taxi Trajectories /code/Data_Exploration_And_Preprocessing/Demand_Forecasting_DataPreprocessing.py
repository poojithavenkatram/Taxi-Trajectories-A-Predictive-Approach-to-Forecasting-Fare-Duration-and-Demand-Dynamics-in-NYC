import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import statistics
import matplotlib.dates as mdates
import time
import datetime

class TaxiDataProcessor:
    def __init__(self, directory_path, shapefile_path):
        self.directory_path = directory_path
        self.shapefile_path = shapefile_path
        self.full_df = None
        self.clean_df = None
        self.filtered_data_2023 = None
        self.location_coords = None

    def load_data(self):
        """Load and concatenate parquet files from a directory."""
        parquet_files = [f for f in os.listdir(self.directory_path) if f.endswith('.parquet')]
        df_list = [pd.read_parquet(os.path.join(self.directory_path, file)) for file in parquet_files]
        self.full_df = pd.concat(df_list, ignore_index=True)

    def preprocess_data(self):
        """Perform initial preprocessing steps on the data."""
        # Dropping unnecessary columns
        cols_to_drop = ['Airport_fee', 'RatecodeID', 'congestion_surcharge', 'VendorID', 'store_and_fwd_flag']
        self.full_df.drop(cols_to_drop, axis=1, inplace=True)
        
        # Fill missing values and drop duplicates
        self.full_df.dropna(subset=['passenger_count'], inplace=True)
        median_airport_fee_count = self.full_df['airport_fee'].median()
        self.full_df['airport_fee'].fillna(self.full_df['airport_fee'].median(), inplace=True)
        self.full_df.drop_duplicates(inplace=True)
        
        # Reset index after dropping rows
        self.full_df.reset_index(drop=True, inplace=True)
        print(self.full_df.shape)
        print(self.full_df)

    def load_shapefile(self):
        """Load shapefile and prepare geographical data."""
        gdf = gpd.read_file(self.shapefile_path)
        gdf['centroid'] = gdf.geometry.centroid
        centroids_geo = gdf['centroid'].to_crs(epsg=4326)
        gdf['latitude'] = centroids_geo.y
        gdf['longitude'] = centroids_geo.x
        self.location_coords = gdf[['LocationID', 'latitude', 'longitude']]



    def add_trip_details_to_df(self):
        def convert_to_unix(s):
            if isinstance(s, datetime.datetime):
                return time.mktime(s.timetuple())
            else:
                return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
        pickup_unix = [convert_to_unix(x) for x in self.full_df['tpep_pickup_datetime']]
        dropoff_unix = [convert_to_unix(x) for x in self.full_df['tpep_dropoff_datetime']]
        self.full_df['trip_times'] = (np.array(dropoff_unix) - np.array(pickup_unix)) / 60.0
        self.full_df['pickup_times'] = pickup_unix
        self.full_df['Speed'] = 60 * (self.full_df['trip_distance'] / self.full_df['trip_times'])

    
    def merge_location_data(self):
        """Merge geographical data with taxi data based on location IDs."""
        if self.location_coords is not None:
            self.full_df = self.full_df.merge(
                self.location_coords, left_on='PULocationID', right_on='LocationID', how='left', suffixes=('', '_pickup')
            )
            self.full_df.rename(columns={'latitude': 'pickup_latitude', 'longitude': 'pickup_longitude'}, inplace=True)

            self.full_df = self.full_df.merge(
                self.location_coords, left_on='DOLocationID', right_on='LocationID', how='left', suffixes=('', '_dropoff')
            )
            self.full_df.drop(['LocationID', 'LocationID_dropoff'], axis=1, inplace=True)
        self.full_df.rename(columns={'latitude': 'dropoff_latitude', 'longitude': 'dropoff_longitude'}, inplace=True)
        print(self.full_df.columns)
        self.full_df.dropna(subset=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], inplace=True)
        print("Mean Passenger Count: ", statistics.mean(self.full_df['passenger_count']))
        print("Median Passenger Count: ", statistics.median(self.full_df['passenger_count']))
        print(pd.crosstab(index = self.full_df['passenger_count'], columns = 'count'))
        self.full_df = self.full_df[(self.full_df['passenger_count'] > 0) & (self.full_df['passenger_count'] <= 6)]

        num_cleaned_entries = self.full_df.shape[0]
        print("Number of entries after removing outliers: ", num_cleaned_entries)

        # Basic statistics
        print(self.full_df['trip_distance'].mean())
        print(self.full_df['trip_distance'].median())
        print(self.full_df['trip_distance'].kurt()) # The higher the kurtosis is often linked to the greater extremity or deviations in the data.
        self.full_df['tpep_pickup_datetime'] = pd.to_datetime(self.full_df['tpep_pickup_datetime'], errors='coerce')
        self.full_df['tpep_dropoff_datetime'] = pd.to_datetime(self.full_df['tpep_dropoff_datetime'], errors='coerce')
        self.add_trip_details_to_df()
        self.full_df.to_csv("../Filtered_Dataset/full_df_before_clean_data.csv",index=False)
        print("Full DF Saved to CSV!!!")

    def plot_data(self):
        """Plot data distributions."""
        sns.countplot(x='passenger_count', data=self.full_df)
        plt.title('Distribution of number of passengers on a trip')
        plt.xlabel('Number of Passengers')
        plt.ylabel('Count')
        plt.show()

        """#Histogram of log-transformed trip distance""" 
        plt.figure(figsize = (4,4))
        sns.displot(np.log(self.full_df['trip_distance'].values+1), bins = 20, aspect = 2, height = 4, color= "black", alpha = 0.6)

        plt.title("Histogram of trip distance on a logarithmic scale", fontsize = 16)
        plt.xlabel("Log-transformed trip distance", fontsize = 14, labelpad = 10)
        plt.ylabel("Frequency", fontsize = 14, labelpad = 10)

        plt.ticklabel_format(style='plain', axis='y')
        plt.savefig('Histogram_trip_distance_log.png', dpi = 300, bbox_inches = 'tight')
        plt.show()
    
    def clean_df_filtering(self):

        self.clean_df = self.full_df[(self.full_df['trip_times'] > 0) & (self.full_df['trip_times'] <= 720)]
        # Confirm removal
        remaining_outliers = self.clean_df[self.clean_df['trip_times'] > 720]
        print("Remaining trips exceeding 720 minutes after re-filtering:", remaining_outliers.shape[0])
        # Ensure that datetime columns are in the correct datetime format
        self.clean_df['tpep_pickup_datetime'] = pd.to_datetime(self.clean_df['tpep_pickup_datetime'], errors='coerce')
        self.clean_df['tpep_dropoff_datetime'] = pd.to_datetime(self.clean_df['tpep_dropoff_datetime'], errors='coerce')

        # Define the start and end date for the year 2023
        start_date = '2023-01-01'
        end_date = '2023-12-31'

        # Filter the data to include only entries within these dates
        self.filtered_data_2023 = self.clean_df[
            (self.clean_df['tpep_pickup_datetime'] >= start_date) &
            (self.clean_df['tpep_pickup_datetime'] <= end_date) &
            (self.clean_df['tpep_dropoff_datetime'] >= start_date) &
            (self.clean_df['tpep_dropoff_datetime'] <= end_date)
        ]

        # Print the number of entries after filtering
        print("Number of entries for the year 2023: ", self.filtered_data_2023.shape[0])
    
    
    

    def EDA(self):
        def time_of_day(x):
            if 6 <= x < 12:
                return 'Morning'
            elif 12 <= x < 16:
                return 'Afternoon'
            elif 16 <= x < 22:
                return 'Evening'
            else:
                return 'Late Night'
        self.filtered_data_2023['tpep_pickup_datetime'] = pd.to_datetime(self.filtered_data_2023['tpep_pickup_datetime'], errors='coerce')
        self.filtered_data_2023['PU_month'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.month.astype(np.uint8)
        self.filtered_data_2023['PU_day_of_month'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.day.astype(np.uint8)
        self.filtered_data_2023['PU_day_of_week'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.weekday.astype(np.uint8)
        self.filtered_data_2023['PU_hour'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.hour.astype(np.uint8)
        PU_day_of_week_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        PU_month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        self.filtered_data_2023['PU_time_of_day'] = self.filtered_data_2023['PU_hour'].apply(time_of_day)
        self.filtered_data_2023['Date'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.date
        date = self.filtered_data_2023['tpep_pickup_datetime'].dt.date
        taxi_group_date = date.groupby(date).agg('count').reset_index(name = 'count')
        taxi_group_date.columns = ['Date', 'Number_of_Pickups']
        taxi_group_date_idx = taxi_group_date.set_index('Date')
        self.filtered_data_2023['Date'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.date
        self.filtered_data_2023 = self.filtered_data_2023.merge(taxi_group_date_idx, on='Date', how='left')
        print(self.filtered_data_2023.head())
        self.filtered_data_2023['tpep_pickup_datetime'] = pd.to_datetime(self.filtered_data_2023['tpep_pickup_datetime'], errors='coerce')
        self.filtered_data_2023['Date'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.date
        self.filtered_data_2023['Month'] = self.filtered_data_2023['tpep_pickup_datetime'].dt.month
        monthly_trips = self.filtered_data_2023.groupby('Month').size().reset_index(name='Number_of_Trips')
        total_trips = monthly_trips['Number_of_Trips'].sum()
        monthly_trips['Percentage'] = (monthly_trips['Number_of_Trips'] / total_trips) * 100
        monthly_trips_check = self.filtered_data_2023.groupby('Month').size().reset_index(name='Correct_Number_of_Trips')
        print(monthly_trips_check)
        self.filtered_data_2023 = self.filtered_data_2023.merge(monthly_trips[['Month', 'Number_of_Trips']], on='Month', how='left')
        self.filtered_data_2023 = self.filtered_data_2023.drop(columns=['Number_of_Trips'], errors='ignore')  # Remove the old column if it exists
        print(self.filtered_data_2023['Month'].unique())
        print(monthly_trips['Month'].unique())
        print(self.filtered_data_2023.duplicated(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance']).sum())
        monthly_trips_corrected = self.filtered_data_2023.groupby('Month').size().reset_index(name='Number_of_Trips')
        self.filtered_data_2023 = self.filtered_data_2023.merge(monthly_trips_corrected, on='Month', how='left')
        


    def save_clean_data(self, filename):
        """Save the cleaned data to a parquet file."""
        self.filtered_data_2023.to_parquet(filename)



# Example usage:
if __name__ == "__main__":
    # Mount Google Drive (specific to Google Colab)
    
    # Create an instance of TaxiDataProcessor
    processor = TaxiDataProcessor(
        directory_path='../yellow_taxi_dataset',
        shapefile_path='../taxi_zones/taxi_zones.shp'
    )
    
    # # Load and preprocess data
    # processor.load_data()
    # print("Data Loaded !!!!")
    # print("Processing Data !!!!")
    # processor.preprocess_data()
    # processor.load_shapefile()
    # print("Merging Location Data !!!!")
    # processor.merge_location_data()
    # print("Plotting Data !!!!")

    processor.full_df = pd.read_csv("../Filtered_Dataset/full_df_before_clean_data.csv")
    # Plotting the data
    # processor.plot_data()
    
    print("Filtering Data for 2023 !!!!")
    processor.clean_df_filtering()
    print("Performing necessary EDA !!!!")
    processor.EDA()
    # Saving cleaned data
    processor.save_clean_data('../Filtered_Dataset/cleaned_taxi_data.parquet')
