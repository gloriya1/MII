import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from beautifultable import BeautifulTable



data = pd.read_csv('train.csv')

print('\n\nИнформация о Dataframe df.info():')
print(data.info())

print('\n\n')
print(data.head(5))
print('\n\n')

X_all = data.iloc[:,:-1].values
y = data['price_range']

# Определение наиболее влияющих признаков
categorial_features = ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory",
                       "m_dep", "mobile_wt", "n_cores", "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
                       "talk_time", "three_g", "touch_screen", "wifi"]

clf = RandomForestClassifier()
clf.fit(X_all, y)
fig, ax = plt.subplots()
ax.barh(categorial_features, clf.feature_importances_)
ax.set_facecolor('seashell')
fig.set_facecolor('floralwhite')
fig.set_figwidth(12)  # ширина Figure
fig.set_figheight(12)  # высота Figure
plt.show()


X = data.iloc[:,:-1].values
X = data[['ram','battery_power', 'px_width','px_height','mobile_wt','int_memory']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


SVC_model = svm.SVC()
KNN_model = KNeighborsClassifier(n_neighbors=5)
TREE_model = DecisionTreeClassifier(max_depth=12)
LOG_model = LinearDiscriminantAnalysis()

SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
TREE_model.fit(X_train,y_train)
LOG_model.fit(X_train,y_train)

SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
TREE_prediction = TREE_model.predict(X_test)
LOG_prediction = LOG_model.predict(X_test)

print("+++++++++++++++++++++KNN_prediction+++++++++++++++++++++")
print(classification_report(KNN_prediction, y_test))
print("+++++++++++++++++++++SVC_prediction+++++++++++++++++++++")
print(classification_report(SVC_prediction, y_test))
print("+++++++++++++++++++++TREE_prediction+++++++++++++++++++++")
print(classification_report(TREE_prediction, y_test))
print("+++++++++++++++++++++LOG_prediction+++++++++++++++++++++")
print(classification_report(LOG_prediction,y_test))

SVC_acc = accuracy_score(SVC_prediction, y_test)
KNN_acc = accuracy_score(KNN_prediction, y_test)
TREE_acc = accuracy_score(TREE_prediction, y_test)
LOG_acc = accuracy_score(LOG_prediction, y_test)



table = BeautifulTable()
table.column_headers = ["Классификатор", "Accuary"]
table.append_row(["Классификатор дерева решений", TREE_acc])
table.append_row(["Линейный дискриминантный анализ", LOG_acc])
table.append_row(["Метод опорных векторов", SVC_acc])
table.append_row(["Метод ближайших соседей", KNN_acc])
print(table)

def tree_vizualization():
    fig = plt.figure(figsize=(200,200))
    _ = tree.plot_tree(TREE_model, feature_names = ['ram','battery_power', 'px_width','px_height','mobile_wt','int_memory'],
                       class_names = ["0","1","2","3"], filled=True)
    fig.savefig("tree.png")

def knn_svm_nai_vizualization(X_test, y_test, point):
    X_test = X_test.values
    y_test = y_test.values

    fig, ax = plt.subplots(4, 1, figsize=(16, 16))

    ax[0].scatter(X_test[:point, 0]+X_test[:point, 1]+X_test[:point, 2],
    X_test[:point, 3]+X_test[:point, 4]+X_test[:point, 5], c=KNN_prediction[:point])
    ax[0].set_title('KNN_prediction')
    ax[0].set_xticks([])

    ax[1].scatter(X_test[:point, 0]+X_test[:point, 1]+X_test[:point, 2],
    X_test[:point, 3]+X_test[:point, 4]+X_test[:point, 5], c=SVC_prediction[:point])
    ax[1].set_title('SVC_prediction')
    ax[1].set_xticks([])

    ax[2].scatter(X_test[:point, 0]+X_test[:point, 1]+X_test[:point, 2],
    X_test[:point, 3]+X_test[:point, 4]+X_test[:point, 5], c=LOG_prediction[:point])
    ax[2].set_title('LOG_prediction')
    ax[2].set_xticks([])

    ax[3].scatter(X_test[:point, 0]+X_test[:point, 1]+X_test[:point, 2],
    X_test[:point, 3]+X_test[:point, 4]+X_test[:point, 5], c=y_test[:point])
    ax[3].set_title('y_test')
    ax[3].set_xticks([])
    plt.show()

knn_svm_nai_vizualization(X_test, y_test, 200)