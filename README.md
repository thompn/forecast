
# Forecast Daily Total Volumes

This code uses the [Prophet](https://facebook.github.io/prophet/) library to forecast daily total volumes using data from a file specified as an argument.

For now, this only covers daily simplistic forecasting.

## Requirements

-   Python 3.6 or higher
-   Pandas
-   Prophet
-   Matplotlib
-   Sklearn

## Usage

First set up your virtual environment. Copy the following code into a terminal:

`python -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

Now make sure you have a csv file in the same directory as the python file. This csv file can include any number of columns, but it **must** include the following columns:
`date` (a column of dates)
`volume_to_predict` (this can have any name, and can be specified in the
 arguments passed when running the python script)
 
__Now you're ready to run, use the following command to run the forecast:__

`python fc.py filename.csv date_column column_to_predict`

Where `filename.csv` is the path to the csv file containing the data to be used to make the forecast, `date_column` is the name of the date column in your input data, and `column_to_predict` is the name of the column containing the values you wish to predict.

For example if your csv file is called `sales.csv` and has the columns `date` and `sales_volume` then you would run as: 

`python fc.py sales.csv date sales_volume`

Alternatively, you can run it more verbosely as follows:

`python forecast.py --filename data.csv --date_column date --column_to_predict volume`

## Output

The code will output two files:

-   `forecast_output_yyyy-mm-dd.csv`: a csv file containing the forecasted total  volumes for each day starting from the day after the last day in the input data.
-   `forecast_yyyy-mm-dd_plots.pdf`: a pdf file containing four plots:
    -   A plot of the actual total volumes used to make the forecast.
    -   A plot of the forecasted total volumes.
    -   A plot showing the components of the forecast, including the trend and weekly and yearly seasonality.
    -  The residuals of the fitted model

The code will also print the mean absolute error (MAE) and mean squared error (MSE) of the forecast to the terminal.