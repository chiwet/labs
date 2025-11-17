import numpy as np
# Генерация случайных температур
temperatures = np.random.randint(-10, 36, size=365)
print(temperatures)
avg_temp = np.average(temperatures)
print(f"Средняя температура в течение года: {avg_temp:.2f}")
lower_mask = temperatures < avg_temp
higher_mask = temperatures > avg_temp
days = np.arange(1, 366)
lower_temp_np = np.column_stack([days[lower_mask], temperatures[lower_mask]])
higher_temp_np = np.column_stack([days[higher_mask], temperatures[higher_mask]])

print(f"\nДни с температурой ниже средней ({avg_temp:.2f}):")
print(lower_temp_np[:])

print(f"\nДни с температурой выше средней ({avg_temp:.2f}):")
print(higher_temp_np[:])

max_temp_day = np.argmax(temperatures)
min_temp_day = np.argmin(temperatures)
print(f"Самая высокая температура: {temperatures[max_temp_day]} (день {max_temp_day + 1})")
print(f"Самая низкая температура: {temperatures[min_temp_day]} (день {min_temp_day + 1})")
jan = np.array(temperatures[0:31])
feb = np.array(temperatures[31:59])
mar = np.array(temperatures[60:91])
apr = np.array(temperatures[91:121])
may = np.array(temperatures[121:152])
jun = np.array(temperatures[152:182])
jul = np.array(temperatures[182:213])
aug = np.array(temperatures[213:244])
sep = np.array(temperatures[244:274])
oct = np.array(temperatures[274:305])
nov = np.array(temperatures[305:335])
dec = np.array(temperatures[335:366])
months_names = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
print(f"Средняя температура в течение года: {avg_temp:.2f}")
months_temp = np.array([jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec],dtype=object)
months_temp_avg = np.array([np.mean(month) for month in months_temp])
print("Средняя температура по месяцам:")
for i, avg_t in enumerate(months_temp_avg):
    print(f"{months_names[i]}: {avg_t:.2f}°C")
spread = 2
print(f"\nНормальные месяцы (Отклонения температуры от средней меньше {spread}°C)")
for i, avg_t in enumerate(months_temp_avg):
    if abs(avg_t - avg_temp) < spread:
        print(f"{months_names[i]}: {avg_t:.2f}°C")
print(f"\nХолодные месяцы (Температура ниже средней на {spread}°C)")
for i, avg_t in enumerate(months_temp_avg):
    if avg_t - avg_temp > spread :
        print(f"{months_names[i]}: {avg_t:.2f}°C")
print(f"\nТеплые месяцы (Температура выше средней на {spread}°C)")
for i, avg_t in enumerate(months_temp_avg):
    if avg_t - avg_temp < spread * -1:
        print(f"{months_names[i]}: {avg_t:.2f}°C")