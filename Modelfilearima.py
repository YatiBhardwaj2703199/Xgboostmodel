import pandas as pd
import numpy as np
import statsmodels.api as sm
from warpdrive import WarpDrive

# Initialize WarpDrive
wd = WarpDrive()

df = wd.get_args("df")

def arima_func(df):
  model = sm.tsa.ARIMA(df.Unemployment_rate, order=(2,3,4)) 
  res_arima = model.fit()
  return res_arima
input_variables = df.drop(columns='Unemployment_rate').columns.tolist()
target_column = 'Unemployment_rate'

# Create and fit ARIMA model
arima_result = arima_func(df)

# Use the correct order and include all parameters
train_table = 'df'  # If no specific table, leave empty or provide a name
lags = 0  # Default lag value
exog_columns = []  # If no exogenous variables

# Create the model
wd.create_model(
    model=arima_result,
    library="statsmodels",
    model_technique="ARIMA",
    input_variables=input_variables,
    target_column=target_column,
    train_table=train_table,
    time_column="Date",
    lags=lags,
    exog_columns=exog_columns
)
