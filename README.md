
# 🌡️ Temperature Forecasting with SARIMA

This is a **Streamlit** web application for forecasting daily average temperatures using the **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) time series model.

## 📌 Features

- Upload a weather CSV file.
- Automatic data cleaning:
  - Date parsing.
  - Fahrenheit to Celsius conversion.
  - Handling missing or duplicate entries.
- SARIMA parameter optimization (auto or manual).
- Model training and forecasting.
- Performance evaluation: **MSE**, **RMSE**, **MAE**, **MAPE**.
- Visualization of actual vs. forecasted temperatures.
- 7-day temperature forecast.

## 🧰 Requirements

- Python 3.7+
- streamlit  
- pandas  
- numpy  
- statsmodels  
- scikit-learn  
- matplotlib

## 🚀 Installation

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Usage

```bash
streamlit run temperature_forecasting_sarima_app.py
```

1. Upload your CSV weather data.
2. Choose automatic or manual SARIMA parameters.
3. View forecasts, metrics, and plots.

## 📁 File Structure

```
├── temperature_forecasting_sarima_app.py  # Main Streamlit app
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

## 📊 Input Format

Your CSV file must contain:
- A `DATE` column in `YYYY-MM-DD` format  
**or**
- Columns: `YEAR`, `MONTH`, `DAY`

Temperature columns can include:
- `TAVG`, `TMAX`, `TMIN` (in Fahrenheit; converted to Celsius automatically)

## 👨‍💻 Team Members

| Name                                 | Academic ID   |
|--------------------------------------|---------------|
| حمزه فارس سمير احمد                 | 412200177     |
| مصطفى أحمد عطية عون                 | 412200378     |
| مصطفى حسين مصطفى صابر              | 412200382     |
| حازم عبدالرحمن فاروق عبدالرحمن      | 412200169     |
| محمد سعيد محمد نصر                  | 412200323     |
| عمر محمد الجبالي السعيد             | 412200269     |

## 📜 License

Licensed under the [MIT License](https://opensource.org/licenses/MIT).
