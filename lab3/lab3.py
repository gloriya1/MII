import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def numpy_analyze():
    with open('data.csv', 'r') as csvFile:
        list_rows = [list(currentRow) for currentRow in csv.reader(csvFile)]

    array_rows = np.array(list_rows)

# Расчеты по дате рождения
    birth =[int(item) for item in array_rows[1:, 3]]
    min_birth = np.min(birth)
    max_birth = np.max(birth)
    sum_birth = np.sum(birth)
    average_birth = np.average(birth)
    disp_birth = np.var(birth)
    standard_deviation_birth= np.std(birth)
    median_birth = np.median(birth)

#Расчеты по зарплате
    salary = [int(item) for item in array_rows[1:, 7]]
    min_salary = np.min(salary)
    max_salary = np.max(salary)
    sum_salary = np.sum(salary)
    average_salary = np.average(salary)
    disp_salary = np.var(salary)
    standard_deviation_salary = np.std(salary)
    median_salary = np.median(salary)

#Расчеты по проектам
    projects = [int(item) for item in array_rows[1:, 8]]
    min_project = np.min(projects)
    max_project = np.max(projects)
    sum_project = np.sum(projects)
    average_project = np.average(projects)
    disp_project = np.var(projects)
    standard_deviation_project = np.std(projects)
    median_project = np.median(projects)



    print('-----------------------------------------------------------------------------------------')
    print('Numpy анализ\n')

    print('Анализ даты рождения:')
    print('Минимальная: ', min_birth)
    print('Максимальная: ', max_birth)
    print('Сумма: ', sum_birth)
    print('Среднее: ', average_birth)
    print('Дисперсия: ', disp_birth)
    print('Стандартное отклонение: ', standard_deviation_birth)
    print('Медиана: ', median_birth)
    print()


    print('Анализ зарплаты:')
    print('Минимальная: ', min_salary)
    print('Максимальная: ', max_salary)
    print('Сумма: ', sum_salary)
    print('Среднее: ', average_salary)
    print('Дисперсия: ', disp_salary)
    print('Стандартное отклонение: ', standard_deviation_salary)
    print('Медиана: ', median_salary)
    print()

    print('Анализ выполненных проектов:')
    print('Минимальное количество: ', min_project)
    print('Максимальное количество: ', max_project)
    print('Сумма: ', sum_project)
    print('Среднее: ', average_project)
    print('Дисперсия: ', disp_project)
    print('Стандартное отклонение: ', standard_deviation_project)
    print('Медиана: ', median_project)
    print()

def pandas_analyze():
    data = pd.read_csv('data.csv', delimiter=',', encoding="windows-1251")

    # Расчеты по дате рождения
    birth = data['Год рождения']
    min_birth = birth.min()
    max_birth = birth.max()
    sum_birth = birth.sum()
    average_birth = birth.mean()
    disp_birth = birth.var()
    standard_deviation_birth = birth.std()
    median_birth = birth.median()

    # Расчеты по зарплате
    salary = data['Оклад']
    min_salary = salary.min()
    max_salary = salary.max()
    sum_salary = salary.sum()
    average_salary = salary.mean()
    disp_salary = salary.var()
    standard_deviation_salary = salary.std()
    median_salary = salary.median()


    # Расчеты по проектам
    projects = data['Количество выполненных проектов']
    min_project = projects.min()
    max_project = projects.max()
    sum_project = projects.sum()
    average_project = projects.mean()
    disp_project = projects.var()
    standard_deviation_project = projects.std()
    median_project = projects.median()


    print('\n------------------------------------------------------------------------------------------')
    print('Pandas анализ\n')

    print('Анализ даты рождения:')
    print('Минимальная: ', min_birth)
    print('Максимальная: ', max_birth)
    print('Сумма: ', sum_birth)
    print('Среднее: ', average_birth)
    print('Дисперсия: ', disp_birth)
    print('Стандартное отклонение: ', standard_deviation_birth)
    print('Медиана: ', median_birth)
    print()


    print('Анализ зарплаты:')
    print('Минимальная: ', min_salary)
    print('Максимальная: ', max_salary)
    print('Сумма: ', sum_salary)
    print('Среднее: ', average_salary)
    print('Дисперсия: ', disp_salary)
    print('Стандартное отклонение: ', standard_deviation_salary)
    print('Медиана: ', median_salary)
    print()

    print('Анализ выполненных проектов:')
    print('Минимальное количество: ', min_project)
    print('Максимальное количество: ', max_project)
    print('Сумма: ', sum_project)
    print('Среднее: ', average_project)
    print('Дисперсия: ', disp_project)
    print('Стандартное отклонение: ', standard_deviation_project)
    print('Медиана: ', median_project)
    print()


def graph():
    data = pd.read_csv('data.csv', delimiter=',', encoding="windows-1251")

    data[['Оклад', 'Должность']].plot(
        kind='scatter',
        x='Оклад',
        y='Должность',
        figsize=(12, 8))

    plt.show()

    plt.bar(data['Начало работы (год)'], data['Количество выполненных проектов'],color = 'yellow', edgecolor='black',
            linewidth=1)
    plt.show()

    eng = 0;
    engi = 0;
    ek = 0;
    engp = 0;
    buh = 0;
    no = 0;
    nt = 0;
    teh = 0;

    for j in data['Должность']:
        if j == 'Инженер': eng = eng + 1;
        if j == 'Инженер-исследователь': engi = engi + 1;
        if j == 'Экономист': ek = ek + 1;
        if j == 'Инженер-программист': engp = engp + 1;
        if j == 'Бухгалтер': buh = buh + 1;
        if j == 'Начальник отдела': no = no + 1;
        if j == 'Начальник ТКБ': nt = nt + 1;
        if j == 'Техник': teh = teh + 1;

    info2 = ['Инженер', 'Инженер-исследователь', 'Экономист', 'Инженер-программист', 'Бухгалтер',
             'Начальник отдела', 'Начальник ТКБ', 'Техник']
    plt.pie([eng, engi, ek, engp, buh, no,nt,teh],labels=info2, autopct='%1.1f%%',startangle=180);
    plt.show()



numpy_analyze()
pandas_analyze()
graph()



