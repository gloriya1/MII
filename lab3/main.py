import csv
import random
from russian_names import RussianNames

#Генерация csv файла
header = [
    'Табельный номер', 'Фамилия И. О.', 'Пол', 'Год рождения', 'Начало работы (год)',
    'Подразделение', 'Должность', 'Оклад', 'Количество выполненных проектов'
]
#количество строк
col = random.Random().randint(1000, 1500)
#col = 5
#print(col)
half_col = int(col/2)
#print(half_col)

year_of_birth = []
year_of_start_career = []
salary =[]
count_project =[]
tabel = []
staff = []
post = []
names = []
gender = [['М']] * half_col + [['Ж']] * (half_col+1)

info1 = ['Информационно-управляющие системы для самолетов', 'Аэрометрические системы воздушных сигналов',
            'Радиоэлектронная аппаратура', 'Информационно-управляющие системы для вертолетов',
            'Проектирование и верификация программно-математического обеспечения',
            'Системы автоматизированного управления наземной техникой', 'Отдел охраны труда', 'Отдел кадров']


info2 = ['Инженер','Инженер-исследователь', 'Экономист', 'Инженер-программист', 'Бухгалтер',
          'Начальник отдела', 'Начальник ТКБ', 'Техник']



row_mens = RussianNames(count=half_col, gender=1.0, patronymic=True,
                        name_reduction=True, patronymic_reduction=True)
row_woman = RussianNames(count=half_col+1, gender=0.0, patronymic=True,
                         name_reduction=True, patronymic_reduction=True)
for i in row_mens:
    names.append([i])
for j in row_woman:
    names.append([j])



for i  in range(col):
    year1 = random.randint(1961,2001)
    year_of_birth.append(year1)
    year_of_start_career.append(year1+20)

    cnt = random.randint(1,20)
    count_project.append(cnt)

    money = random.randint(20000, 60000)
    salary.append(money)

    tabel.append(i)

    depart = random.choice(info1)
    staff.append(depart)

    rang = random.choice(info2)
    post.append(rang)


#запись данных в csv файл
with open('data.csv','w', newline='') as f:
    # delimiter=";"
    write = csv.DictWriter(f, fieldnames = header)
    write.writeheader()
    for i in range(col):
        write.writerow({'Табельный номер': tabel[i], 'Фамилия И. О.': ','.join(names[i]),
                        'Пол': ','.join(gender[i]), 'Год рождения': year_of_birth[i],
                        'Начало работы (год)': year_of_start_career[i],
                        'Подразделение': staff[i],'Должность': post[i],
                        'Оклад': salary[i],'Количество выполненных проектов': count_project[i]})

