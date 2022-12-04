import unittest
import warnings
warnings.simplefilter('ignore')

import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import mean_absolute_error, mean_squared_error

class ProphetTests(unittest.TestCase):

    def test_data_read(self):
        # read the file into a df
        with open(args.filename, 'r') as f:
            df = pd.read_csv(f)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0)
        self.assertGreater(df.shape[1], 0)

    def test_date_column_transform(self):
        # transform the date column to a datetime type
        df['date'] = pd.to_datetime(df['date'])
        self.assertIsInstance(df['date'], pd.DatetimeIndex)

    def test_aggregation(self):
        # group by the date, to aggregate daily
        # and sum the total volumes to get a daily aggregated
        # call volume, ignoring the country split
        df = df.groupby(args.date_column).agg({args.column_to_predict:'sum'}).reset_index()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0)
        self.assertGreater(df.shape[1], 0)
        
    def test_rename_columns(self):
        # rename the columns to be accepted by prophet
        df.columns = ['ds','y']
        self.assertEqual(df.columns[0], 'ds')
        self.assertEqual(df.columns[1], 'y')

    def test_instantiate_prophet(self):
        # instantiate prophet
        m = Prophet()
        self.assertIsNotNone(m)
        self.assertIsInstance(m, Prophet)

    def test_fit_model(self):
        # fit the model to the call data in df
        model = m.fit(df)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, Prophet)

    def test_make_future_dataframe(self):
        # use the fitted model to make a prediction +365 periods on a daily basis
        future = m.make_future_dataframe(periods=365, freq='D')
        self.assertIsNotNone(future)
        self.assertIsInstance(future, pd.DataFrame)
        self.assertGreater(future.shape[0], 0)
        self.assertGreater(future.shape[1], 0)
        
    def test_predict_forecast(self):
        # predict the forecast and save to forecast df
        forecast = m.predict(future)
        self.assertIsNotNone(forecast)
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertGreater(forecast.shape[0], 0)
        self.assertGreater(forecast.shape[1], 0)

    def test_concat_dataframes(self):
        # Combine the two dataframes using the 'ds' column as the key
        df_combined = pd.concat([df, forecast], axis=1, join='inner')
        self.assertIsNotNone(df_combined)
        self.assertIsInstance(df_combined, pd.DataFrame)
        self.assertGreater(df_combined.shape[0], 0)
        self.assertGreater(df_combined.shape[1], 0)
        
    def test_calculate_residuals(self):
        # Calculate the residuals by subtracting the predicted values from the actual values
        residuals = df_combined['y'] - df_combined['yhat']
        self.assertIsNotNone(residuals)
        self.assertIsInstance(residuals, pd.Series)
        self.assertGreater(residuals.shape[0], 0)
        
    def test_calculate_metrics(self):
        # Calculate the MAE and MSE
        mae = mean_absolute_error(df_combined['y'], df_combined['yhat'])
        mse = mean_squared_error(df_combined['y'], df_combined['yhat'])
        self.assertIsNotNone(mae)
        self.assertIsNotNone(mse)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(mse, float)
        
    def test_save_forecast_output(self):
        # filter only date and volumes to output
        to_save = forecast[['ds','yhat']]
        to_save.columns = ['date','forecast_value']

        # start forecast output from tomorrow
        forecast_ready = to_save[to_save['date'] > tddt]

        # save forecast output to csv file
        forecast_ready.to_csv(f'forecast_output_{tddt}.csv')
        self.assertIsNotNone(forecast_ready)
        self.assertIsInstance(forecast_ready, pd.DataFrame)
        self.assertGreater(forecast_ready.shape[0], 0)
        self.assertGreater(forecast_ready.shape[1], 0)

if __name__ == '__main__':
    unittest.main()