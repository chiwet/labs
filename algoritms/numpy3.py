import numpy as np
sales = np.array([
[120, 340, 560, 230], # Январь
[150, 400, 600, 280], # Февраль
[180, 390, 630, 310], # Март
[170, 420, 670, 290], # Апрель
[200, 450, 710, 330], # Май
[220, 470, 750, 350], # Июнь
])
months = np.array(["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь"])
categories = np.array(["Электроника", "Одежда", "Бытовая техника", "Мебель"])
print(f"1. Объём продаж за 6 месяцев: {np.sum(sales)}")
print(f"2. Месяц с наибольшими продажами: {months[np.argmax(np.sum(sales, axis=1))]}")
print(f"3. Товар с наибольшими продажами: {categories[np.argmax(np.sum(sales, axis=0))]} ({np.sum(sales, axis=0)[np.argmax(np.sum(sales, axis=0))]})")
print(f"""4. Средний объём продаж каждой категории товаров:\n
        {categories[0]} - {np.average(sales, axis=0)[0]:.2f}
        {categories[1]} - {np.average(sales, axis=0)[1]:.2f}
        {categories[2]} - {np.average(sales, axis=0)[2]:.2f}
        {categories[3]} - {np.average(sales, axis=0)[3]:.2f}
""")
print(f"""5. Рост продаж за первый и последний месяц для каждой категории:
        {categories[0]} - {(sales[5,0] / sales[0,0] - 1) * 100:.2f}%
        {categories[1]} - {(sales[5,1] / sales[0,1] - 1) * 100:.2f}%
        {categories[2]} - {(sales[5,2] / sales[0,2] - 1) * 100:.2f}%
        {categories[3]} - {(sales[5,3] / sales[0,3] - 1) * 100:.2f}%
      """)