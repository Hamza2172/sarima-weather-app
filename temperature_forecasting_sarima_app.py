import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def fahrenheit_to_celsius(temp_f):
    return (temp_f - 32) * 5/9

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    # Construct date from YEAR, MONTH, DAY if DATE not present
    if 'DATE' in data.columns:
        data['DATE'] = pd.to_datetime(data['DATE'])
    elif {'YEAR', 'MONTH', 'DAY'}.issubset(set(data.columns)):
        data['DATE'] = pd.to_datetime(data[['YEAR', 'MONTH', 'DAY']])
    else:
        st.error("CSV file must contain 'DATE' or 'YEAR', 'MONTH', 'DAY' columns.")
        return None
    data.set_index('DATE', inplace=True)
    # Remove duplicate dates keeping the first occurrence
    data = data[~data.index.duplicated(keep='first')]
    temp_cols = ['TAVG', 'TMAX', 'TMIN']
    for col in temp_cols:
        if col in data.columns:
            data[col] = data[col].apply(fahrenheit_to_celsius)
    if 'TAVG' in data.columns and 'TMAX' in data.columns and 'TMIN' in data.columns:
        data['TAVG'] = data.apply(
            lambda row: (row['TMAX'] + row['TMIN']) / 2 if pd.isnull(row['TAVG']) and not pd.isnull(row['TMAX']) and not pd.isnull(row['TMIN']) else row['TAVG'],
            axis=1
        )
    data = data.dropna(subset=['TAVG'])
    data = data.resample('D').mean(numeric_only=True)
    return data['TAVG']

def find_best_sarima_params(data, max_p=4, max_d=2, max_q=4, max_P=2, max_D=1, max_Q=2, m=7):
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    for p in range(max_p+1):
        for d in range(max_d+1):
            for q in range(max_q+1):
                for P in range(max_P+1):
                    for D in range(max_D+1):
                        for Q in range(max_Q+1):
                            try:
                                model = SARIMAX(data, order=(p,d,q), seasonal_order=(P,D,Q,m), enforce_stationarity=False, enforce_invertibility=False)
                                results = model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p,d,q)
                                    best_seasonal_order = (P,D,Q,m)
                            except:
                                continue
    return best_order, best_seasonal_order

def train_sarima_model(data, order, seasonal_order):
    train_size = int(len(data)*0.8)
    train, test = data[:train_size], data[train_size:]
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_values = forecast.predicted_mean
    return model_fit, train, test, forecast_values

def calculate_metrics(y_true, y_pred):
    # Drop NaN values from y_true and y_pred for metric calculation
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mask_mape = np.abs(y_true_clean) > 1e-3
    if np.any(mask_mape):
        mape = np.mean(np.abs((y_true_clean[mask_mape] - y_pred_clean[mask_mape]) / y_true_clean[mask_mape])) * 100
    else:
        mape = float('nan')
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def plot_forecast(train, test, forecast):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('SARIMA Temperature Forecast')
    plt.legend()
    st.pyplot(plt)

def main():
    st.title("🌡️ Temperature Forecasting with SARIMA")
    st.write("This app forecasts daily average temperature using SARIMA model.")
    
    uploaded_file = st.file_uploader("Upload your weather CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_and_prepare_data(uploaded_file)
        if data is None:
            return
        st.write(f"Data loaded: {len(data)} records from {data.index.min().date()} to {data.index.max().date()}")
        st.write("Sample data:")
        st.write(data.head(10))
        
        if st.button("Find Best SARIMA Parameters and Train Model"):
            with st.spinner("Training SARIMA model..."):
                best_order, best_seasonal_order = find_best_sarima_params(data)
                st.write(f"Best SARIMA order found: {best_order} Seasonal order: {best_seasonal_order}")
                model_fit, train, test, forecast = train_sarima_model(data, best_order, best_seasonal_order)
                st.write("Sample test data:")
                st.write(test.head(10))
                st.write("Sample forecast data:")
                st.write(forecast.head(10))
                metrics = calculate_metrics(test, forecast)
                st.write("Model Performance Metrics:")
                st.write(metrics)
                forecast.index = test.index
                plot_forecast(train, test, forecast)
                
                # Forecast next 7 days
                forecast_horizon = 7
                future_forecast = model_fit.forecast(steps=forecast_horizon)
                future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_horizon)
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast Temperature (°C)': future_forecast.values})
                st.subheader("Forecast for Next 7 Days")
                st.write(forecast_df)
    else:
        st.info("Please upload a CSV file to start.")

if __name__ == "__main__":
    main()
