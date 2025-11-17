import numpy as np
sales = np.random.randint(500, 1500, (3, 12)) # Данные о продажах (случайные числа)
years = ['2022', '2023', '2024']
months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
print(sales)
sales_avg = np.average(sales,axis=1)


print(f"Средние продажи за каждый год: 2022 - {sales_avg[0]:.2f}, 2023 - {sales_avg[1]:.2f}, 2024 - {sales_avg[2]:.2f}")
monthly_avg = np.average(sales, axis=0)
deviations = sales - monthly_avg
max_deviation = np.max(deviations)
max_deviation_idx = np.unravel_index(np.argmax(deviations), deviations.shape)
print(f"\nМаксимальный прирост продаж:")
print(f"Год: {years[max_deviation_idx[0]]}")
print(f"Месяц: {months[max_deviation_idx[1]]}")
print(f"Продажи: {sales[max_deviation_idx]} единиц")
print(f"Среднее для этого месяца: {monthly_avg[max_deviation_idx[1]]:.2f} единиц")
print(f"Прирост: +{max_deviation:.2f} единиц")



X = np.arange(sales.size).reshape(-1, 1)
Y = sales.flatten()

X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

coefficients, residuals, rank, s = np.linalg.lstsq(X_with_intercept, Y, rcond=None)
intercept, slope = coefficients

print(f"\n\nУравнение регрессии: y = {slope:.2f}x + {intercept:.2f}")

next_year_start = sales.size
next_year_months = np.arange(next_year_start, next_year_start + 12).reshape(-1, 1)
next_year_X = np.column_stack([np.ones(next_year_months.shape[0]), next_year_months])
predicted_sales = next_year_X @ coefficients
print("\nПрогноз продаж на 2025 год:")
print(np.array(predicted_sales))