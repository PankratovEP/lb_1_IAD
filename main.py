import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sb
import pingouin as pg
from scipy.stats import kstest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import csv


iris = load_iris()


# датафрейм имеет вид таблицы со столбами: 'sepal length (cm)' , 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data']], columns=iris['feature_names'])
undepend = np.array(iris_pd['petal length (cm)']).reshape((-1, 1)) # зависимая выходит
depended = (iris_pd['sepal length (cm)'], iris_pd['sepal width (cm)'], iris_pd['petal width (cm)']) # независимая входит предикаторы
chemic_bd = pd.read_csv('Chemical_process.csv', delimiter=';', dtype='float64')



def box_plots():
    for caharateristic in iris_pd:
        print(iris_pd[caharateristic].describe(), end='\n*****************\n')
    sb.boxplot(data=iris_pd)
    plt.show()


# .corr()-метод для парных коэфов
def heat_map(dat=iris_pd):
    sb.heatmap(dat.corr(), annot=True, cmap='coolwarm', linewidths=3, linecolor='black')
    plt.show()


def partial_heat(da=iris_pd):
    dat = pg.pcorr(da)
    sb.heatmap(data=dat, annot=True, cmap='coolwarm', linewidths=3, linecolor='black')
    plt.show()


# H0- норм распределение alpha = 0.05
def kolmogorov_check():
    for i in iris_pd:
        print(i, kstest(iris_pd[i], 'norm'), sep='\n')


def histograms():
    plt.subplot(2, 2, 1)
    plt.hist(iris_pd['sepal length (cm)'], bins=20)
    plt.title('Длина Чашелистика')
    plt.subplot(2, 2, 2)
    plt.hist(iris_pd['sepal width (cm)'], bins=20)
    plt.title('Ширина Чашелистика')
    plt.subplot(2, 2, 3)
    plt.hist(iris_pd['petal length (cm)'], bins=20)
    plt.title('Длина Лепестка')
    plt.subplot(2, 2, 4)
    plt.hist(iris_pd['petal width (cm)'], bins=20)
    plt.title('Ширина лепестка')
    plt.show()


def qqplots():
    pg.qqplot(iris_pd['sepal length (cm)'])
    plt.title('sepal length (cm)')
    pg.qqplot(iris_pd['sepal width (cm)'])
    plt.title('sepal width (cm)')
    pg.qqplot(iris_pd['petal length (cm)'])
    plt.title('petal length (cm)')
    pg.qqplot(iris_pd['petal width (cm)'])
    plt.title('petal width (cm)')
    plt.show()


def scat_plot():
    plt.subplot(2, 2, 1)
    plt.scatter(iris_pd['petal length (cm)'], iris_pd['sepal length (cm)'])
    plt.title('petal length(x) - sepal length')
    plt.subplot(2, 2, 2)
    plt.scatter(iris_pd['petal length (cm)'], iris_pd['sepal width (cm)'])
    plt.title('petal length(x) - sepal width')
    plt.subplot(2, 2, 3)
    plt.scatter(iris_pd['petal length (cm)'], iris_pd['petal width (cm)'])
    plt.title('petal length(x) - petal width')
    plt.show()


# Господь всемогущий покинул эту функцию на костылях
def mozh_regr():
    X = iris_pd.drop(columns=['petal length (cm)', 'sepal width (cm)'])
    y = iris_pd['petal length (cm)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeffs'])
    print(coef_df)
    print(f'intercept {model.intercept_}')
    print(f'coef determination {model.score(X, y)}')
    det = model.score(X, y)
    print(f'mulipile corr coef {model.score(X, y) ** 0.5}')
    print(f'Скорректированный кф детерминации {1 - (1 - det)*(149 / (150 - 2 - 1))}')
    y_pred = model.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    reg_m = LinearRegression().fit(iris_pd.drop(columns=['petal length (cm)', 'sepal width (cm)']), iris_pd['petal length (cm)'])
    prediction = reg_m.predict(iris_pd.drop(columns=['petal length (cm)', 'sepal width (cm)']))
    y = []
    for i in range(len(prediction)):
        y.append(model.intercept_ + 0.5598*iris_pd['sepal length (cm)'][i] + 1.7772*iris_pd['petal width (cm)'][i])
    y.sort()
    real = sorted(iris_pd['petal length (cm)'])
    ls = [(real[i] - y[i]) for i in range(len(prediction))]
    def one_million_graphiks():
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('predicted_pet_len')
        ax.set_ylabel('sepal length (cm)')
        ax.set_zlabel('petal width (cm)')
        sequence_containing_x_vals = prediction
        sequence_containing_y_vals = iris_pd['sepal length (cm)']
        sequence_containing_z_vals = iris_pd['petal width (cm)']
        ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
        plt.title('PREDICTION')
        plt.show()
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('petal length (cm)')
        ax.set_ylabel('sepal length (cm)')
        ax.set_zlabel('petal width (cm)')
        sequence_containing_x_vals = iris_pd['petal length (cm)']
        sequence_containing_y_vals = iris_pd['sepal length (cm)']
        sequence_containing_z_vals = iris_pd['petal width (cm)']
        ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
        plt.title('REAL')
        plt.show()
        plt.scatter(iris_pd['petal length (cm)'], ls)
        plt.plot(undepend, [0] * 150, color='Red')
        plt.title('График остатков')
        plt.show()
        pg.qqplot(ls)
        plt.show()
    print(f'MAE = {(1/150)*sum([abs(iris_pd["petal length (cm)"][i] - prediction[i]) for i in range(len(prediction))])}')
    print(kstest(ls, 'norm'), sep='\n')
    def histr():
        plt.hist(ls, bins=17)
        plt.show()
    print(f'Durbin_Watson : {durbin_watson(ls)}')


#парная регрессия рабочая вход - петал  ленх
def isprav(vhod, vihod):
    model = LinearRegression()
    X = pd.DataFrame(iris_pd[vhod])
    y = pd.DataFrame(iris_pd[vihod])
    model.fit(X, y)
    det_cf = model.score(X, y)
    print(f'b coef: {model.coef_}')
    print(f'intercept: {model.intercept_}')
    print(f'Coef determination: {det_cf}')
    print(f'MAE: {mean_absolute_error(y_true=y, y_pred=model.predict(X))}')
    df_ostatok = y - model.predict(X)
    def parniy_graph():
        plt.scatter(iris_pd[vhod], iris_pd[vihod])
        plt.plot(X, model.predict(X), color='Red', label='Regression line')
        plt.axhline(y=0, color='orange', linestyle='--', linewidth=1)
        plt.scatter(y, df_ostatok, color='green', label='Остатки')
        plt.xlim(-10, 10)
        plt.ylim(-5, 9)
        plt.legend()
        plt.grid()
        plt.xlabel(vhod)
        plt.ylabel(vihod)
        plt.show()
    def zavis_ot_n ():
        plt.scatter(list(range(150)), y, color='blue',  label='реальные')
        plt.scatter(list(range(150)), model.predict(X), color='Red', label='предсказанные')
        plt.xlabel('Номер в выборке')
        plt.ylabel(vihod)
        plt.grid()
        plt.legend()
        plt.show()



def model_check():
    X = iris_pd.drop(columns=['petal length (cm)', 'sepal width (cm)'])
    y = iris_pd['petal length (cm)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Тренировочная  модель
    model = LinearRegression()
    model.fit(X_train, y_train)
    coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeffs'])


    def train():
        print('        TRAIN')
        print(coef_df)
        print(f'intercept: {model.intercept_}')
        print(f'coef determination: {model.score(X_train, y_train)}')
        print(f'Скорректированный кф детерминации {1 - (1 - model.score(X_train, y_train))*(119 / (120 - 2 - 1))}')
        print(f'coef correlation: {model.score(X_train, y_train) ** 0.5}')
        print(f'MSE : {mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))}')
        return plt.scatter(y_train, y_train - model.predict(X_train))

    def tesst():
        print('\n        TEST')
        print(coef_df)
        print(f'intercept: {model.intercept_}')
        print(f'coef determination: {model.score(X_test, y_test)}')
        print(f'Скорректированный кф детерминации {1 - (1 - model.score(X_test, y_test)) * (29 / (30 - 2 - 1))}')
        print(f'coef correlation: {model.score(X_test, y_test) ** 0.5}')
        print(f'MSE : {mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))}')
        return plt.scatter(y_test, y_test - model.predict(X_test))

    plt.subplot(1, 2, 1)
    train()
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.title('Остатки от тренировочной')
    plt.subplot(1, 2, 2)
    tesst()
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.title('Остатки от тестовой')
    plt.show()


def ispr_mnozh():
    X = iris_pd.drop(columns=['petal length (cm)', 'sepal width (cm)'])
    y = iris_pd['petal length (cm)']
    model = LinearRegression()
    model.fit(X, y)
    determin = model.score(X, y)
    coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeffs'])
    y_preds = model.predict(X)
    print(coef_df)
    print(f'intercept: {model.intercept_}')
    print(f'determin coeff: {determin}')
    print(f'mulipile corr coef {determin ** 0.5}')
    print(f'Скорректированный кф детерминации {1 - (1 - determin)*(149 / (150 - 2 - 1))}')
    print(f'MAE: {mean_absolute_error(y_true=y, y_pred=y_preds)}')
    ostatki = y - y_preds
    def graphiki():
        plt.subplot(2, 3, 1)
        plt.scatter(X['sepal length (cm)'], y_preds)
        plt.title('Предсказание по sepal length (cm)-х')
        plt.subplot(2, 3, 2)
        plt.scatter(X['petal width (cm)'], y_preds)
        plt.title('Предсказание по petal width (cm)-х')
        plt.subplot(2, 3, 3)
        plt.scatter(y, ostatki)
        plt.axhline(y=0, linewidth=1, color='red')
        plt.title('График остатков')
        plt.subplot(2, 3, 4)
        plt.scatter(X['sepal length (cm)'], y)
        plt.title('реальное распред. по sepal length (cm)-х')
        plt.subplot(2, 3, 5)
        plt.scatter(X['petal width (cm)'], y)
        plt.title('реальное распред. по petal width (cm)-х')
        plt.subplot(2, 3, 6)
        pg.qqplot(ostatki)
        plt.show()
    graphiki()

ispr_mnozh()

