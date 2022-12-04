import sys
import os
sys.path.append("../")

import unittest
from fc import Forecaster
import pandas as pd
from prophet import Prophet
from datetime import datetime as dt

#assign today's date to a variable
tddt = dt.strftime(dt.today(), '%Y-%m-%d')

class TestForecaster(unittest.TestCase):

    def setUp(self):
        self.filename = '../calls.csv'
        self.date_column = 'date'
        self.column_to_predict = 'total_inbound_calls'
        self.forecaster = Forecaster(self.filename, self.date_column, self.column_to_predict)

    def test_read_file(self):
        df = self.forecaster.read_file()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_transform_data(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_fit_model(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, Prophet)

    def test_predict_model(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        forecast = self.forecaster.predict_model(model)
        self.assertIsNotNone(forecast)
        self.assertIsInstance(forecast, pd.DataFrame)

    def test_combine_dataframes(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        forecast = self.forecaster.predict_model(model)
        df_combined = self.forecaster.combine_dataframes(df, forecast)
        self.assertIsNotNone(df_combined)
        self.assertIsInstance(df_combined, pd.DataFrame)

    def test_calculate_residuals(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        forecast = self.forecaster.predict_model(model)
        df_combined = self.forecaster.combine_dataframes(df, forecast)
        residuals = self.forecaster.calculate_residuals(df_combined)
        self.assertIsNotNone(residuals)
        self.assertIsInstance(residuals, pd.Series)

    def test_calculate_metrics(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        forecast = self.forecaster.predict_model(model)
        df_combined = self.forecaster.combine_dataframes(df, forecast)
        mae, mse = self.forecaster.calculate_metrics(df_combined)
        self.assertIsNotNone(mae)
        self.assertIsNotNone(mse)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(mse, float)

    def test_format_forecast_output(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        forecast = self.forecaster.predict_model(model)
        forecast_ready = self.forecaster.format_forecast_output(forecast)
        self.assertIsNotNone(forecast_ready)
        self.assertIsInstance(forecast_ready, pd.DataFrame)

    def test_save_forecast_output(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        forecast = self.forecaster.predict_model(model)
        forecast_ready = self.forecaster.format_forecast_output(forecast)
        self.forecaster.save_forecast_output(forecast_ready)
        self.assertTrue(os.path.exists(f'forecast_output_{tddt}'))

    def test_save_plots(self):
        df = self.forecaster.read_file()
        df = self.forecaster.transform_data(df)
        model = self.forecaster.fit_model(df)
        forecast = self.forecaster.predict_model(model)
        df_combined = self.forecaster.combine_dataframes(df, forecast)
        residuals = self.forecaster.calculate_residuals(df_combined)
        forecast_ready = self.forecaster.format_forecast_output(forecast)
        self.forecaster.save_plots(model, forecast, forecast_ready, residuals)
        self.assertTrue(os.path.exists(f'forecast_{tddt}_plots.pdf'))

if __name__ == '__main__':
    unittest.main()