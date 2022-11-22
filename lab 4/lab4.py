from math import sqrt
import csv
from matplotlib import pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------------------------------------------
#Метрический классификатор

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2

    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])

    neighbors = list()
    for i in range(num_neighbors):

        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
# Классификатор sklearn

def knn_sklearn(points, y, k):
    point_train, point_test, class_train, class_test = train_test_split(points,
    y,test_size=0.2, shuffle=False, stratify=None)
    scaler=StandardScaler()
    scaler.fit(point_train)

    point_train=scaler.transform(point_train)
    point_test=scaler.transform(point_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(point_train, class_train)
    prognoz=model.predict(point_test)
    return point_train, point_test, class_train, class_test, prognoz
#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
# Генерация данных

title=["продукт","сладость","хруст","класс"]
datasetbig = [
            ['яблоко', 7, 7, 0],
            ['салат', 2, 5, 1],
            ['бекон', 1, 2, 2],
            ['орехи', 1, 5, 2],
            ['рыба', 1, 1, 2],
            ['хурма',8, 2, 0],
            ['банан', 9, 1, 0],
            ['морковь',	2, 8, 1],
            ['виноград', 8, 1, 0],
            ['апельсин', 6, 1, 0],
            ['клубника', 3, 7, 1],
            ['шашлык', 1, 1, 2],
            ['груша', 5, 3, 0],
            ['сельдерей', 1, 5, 1],
            #train data - 14, test data - 4
            ['арбуз', 5, 3, 0],
            ['сыр', 1, 1, 2],
            ['редиска',0, 9, 1],
            ['манго',8, 3, 0]]

with open("data1.csv", mode="w") as wr_file:
    writer=csv.writer(wr_file, delimiter=",", lineterminator="\n")
    writer.writerow(title)
    for i in range(0,18):
        item=datasetbig[i]
        writer.writerow(item)
with open("data1.csv", mode="r") as r_file:
    data=[]
    dataset = []
    reader=csv.reader(r_file, delimiter=",",lineterminator="\n")
    flag = 0
    for row in reader:
        if flag != 0:
            data.append([row[0], int(row[1]), int(row[2]), int(row[3])])
            dataset.append([int(row[1]), int(row[2]), int(row[3])])
        else: flag+=1

testdata = dataset[14:]

#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
# Вывод
print("Метрический классификатор")
for i in range(len(testdata)):
    prediction = predict_classification(dataset, testdata[i], 4)
    print("Продукт",  data[14+i][0])
    print('Реальный класс %d, Полученный класс %d.' % (testdata[i][-1], prediction))
#-----------------------------------------------------------------------------------------------------------------
print("\nКлассификатор sklearn")
points=[]
classes=[]
for i in range(len(dataset)):
    points.append([dataset[i][0],dataset[i][1]])
    classes.append(dataset[i][2])
point_train, point_test, classes_train, classes_test, prognoz=knn_sklearn(points, classes, 4)
print("По версии sklearn продукты относятся к классам: ", prognoz)
print("Правильный ответ: ", classes_test)
#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
# ДОБАВЛЕНИЕ НОВОГО КЛАССА
#-----------------------------------------------------------------------------------------------------------------
# Генерация данных

title2=["продукт","сладость","хруст","класс"]
# 0 - фрукты, 1 - овощи, 2 - мясо, 3 - молочное
datasetbig2 = [
            ['яблоко', 7, 7, 0],
            ['салат', 2, 5, 1],
            ['бекон', 1, 2, 2],
            ['орехи', 1, 5, 2],
            ['рыба', 1, 1, 2],
            ['сливки', 5, 0, 3],
            ['ряженка', 3, 0, 3],
            ['хурма',8, 2, 0],
            ['банан', 9, 1, 0],
            ['морковь',	2, 8, 1],
            ['виноград', 8, 1, 0],
            ['апельсин', 6, 1, 0],
            ['кефир', 2, 0, 3],
            ['творог', 4, 2, 3],
            ['шашлык', 1, 1, 2],
            ['сельдерей', 1, 5, 1],
            #train data - 16, test data - 5
            ['креветки', 2, 2, 2],
            ['груша', 6, 5, 0],
            ['сыр', 3, 2, 3],
            ['редиска',0, 9, 1],
            ['манго',8, 3, 0]]

with open("data2.csv", mode="w") as wr_file:
    writer2=csv.writer(wr_file, delimiter=",", lineterminator="\n")
    writer2.writerow(title2)
    for i in range(0,21):
        item2=datasetbig2[i]
        writer2.writerow(item2)
with open("data2.csv", mode="r") as r_file:
    data2=[]
    dataset2 = []
    reader2=csv.reader(r_file, delimiter=",",lineterminator="\n")
    flag2 = 0
    for row in reader2:
        if flag2 != 0:
            data2.append([row[0], int(row[1]), int(row[2]), int(row[3])])
            dataset2.append([int(row[1]), int(row[2]), int(row[3])])
        else: flag2+=1

testdata2 = dataset2[16:]

#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
# Вывод

print("\nМетрический классификатор 2")
for i in range(len(testdata2)):
    prediction2 = predict_classification(dataset2, testdata2[i], 3)
    print("Продукт",  data2[16+i][0])
    print('Реальный класс %d, Полученный класс %d.' % (testdata2[i][-1], prediction2))
#-----------------------------------------------------------------------------------------------------------------
print("\nКлассификатор sklearn 2")
points2=[]
classes2=[]
for i in range(len(dataset2)):
    points2.append([dataset2[i][0],dataset2[i][1]])
    classes2.append(dataset2[i][2])
point_train2, point_test2, classes_train2, classes_test2, prognoz2=knn_sklearn(points2, classes2, 3)
print("По версии sklearn продукты относятся к классам: ", prognoz2)
print("Правильный ответ: ", classes_test2)
#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
# Визуализация

def showData (trainData, testData):
    classColormap  = ListedColormap(['#FF0000', '#00FF00','#FFA500', '#0000FF'])
    pl.scatter([trainData[i][0] for i in range(len(trainData))],
               [trainData[i][1] for i in range(len(trainData))],
               c=[trainData[i][2] for i in range(len(trainData))],
               cmap=classColormap)
    pl.scatter([testData[i][0] for i in range(len(testData))],
               [testData[i][1] for i in range(len(testData))],
               c=[testData[i][2] for i in range(len(testData))],
               cmap=classColormap)
    for i in range(len(testData)):
        if trainData[i][2] != 2:
            pl.text(x=testData[i][0] - 1.7, y=testData[i][1] + 0.5, s=f"new point, class: {testData[i][2]}")
        else:
            pl.text(x=testData[i][0] - 1.7, y=testData[i][1] - 0.5, s=f"new point, class: {testData[i][2]}")
    pl.show()

showData(dataset,testdata)
showData(dataset2,testdata2)