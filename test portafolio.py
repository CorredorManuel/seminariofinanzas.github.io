import yfinance as yf
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay

# Configuración
symbol = "BNS"
earnings_dates = [
    "2020-12-01", "2021-02-23", "2021-06-01", "2021-08-24",
    "2021-11-30", "2022-03-01", "2022-05-25", "2022-08-23",
    "2022-11-29", "2023-02-28", "2023-05-24"
]

percentage_changes = []  # Lista para almacenar las variaciones porcentuales

for earnings_date_str in earnings_dates:
    earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
    end_date = earnings_date + BDay(5)
    
    data = yf.download(symbol, start=earnings_date, end=end_date)
    
    initial_price = data["Close"].iloc[0]
    final_price = data["Close"].iloc[-1]
    
    percentage_change = ((final_price - initial_price) / initial_price) * 100
    percentage_changes.append(percentage_change)
    
    print(f"Fecha de reporte: {earnings_date_str}")
    print(f"Fecha de CIERRE: {end_date}")
    print(f"Precio inicial: {initial_price:.2f}")
    print(f"Precio final: {final_price:.2f}")
    print(f"Variación porcentual: {percentage_change:.2f}%")
    print("------------------------")

# Calcular el promedio de las variaciones porcentuales
average_percentage_change = sum(percentage_changes) / len(percentage_changes)

print(f"Promedio de variaciones porcentuales: {average_percentage_change:.2f}%")
