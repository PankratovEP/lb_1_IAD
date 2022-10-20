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

iris = load_iris()


# датафрейм имеет вид таблицы со столбами: 'sepal length (cm)' , 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data']], columns=iris['feature_names'])
undepend = np.array(iris_pd['petal length (cm)']).reshape((-1, 1)) # зависимая выходит
depended = (iris_pd['sepal length (cm)'], iris_pd['sepal width (cm)'], iris_pd['petal width (cm)']) # независимая входит предикаторы


def box_plots():
    for caharateristic in iris_pd:
        print(iris_pd[caharateristic].describe(), end='\n*****************\n')
    sb.boxplot(data=iris_pd)
    plt.show()


# .corr()-метод для парных коэфов
def heat_map():
    sb.heatmap(data=iris_pd.corr(), annot=True, cmap='coolwarm', linewidths=3, linecolor='black')
    plt.show()


def heat_map_p():
    sb.heatmap(data=iris_pd.pcorr(), annot=True, cmap='coolwarm', linewidths=3, linecolor='black')
    plt.show()


def partial_heat():
    dat = pg.pcorr(iris_pd)
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


def threed_scatter():
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')
    sequence_containing_x_vals = iris_pd['petal length (cm)']
    sequence_containing_y_vals = iris_pd['sepal length (cm)']
    sequence_containing_z_vals = iris_pd['petal width (cm)']
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    plt.show()


def par_regr_petlen_seplen():
    model = LinearRegression().fit(undepend, depended[0])
    r_sq = model.score(undepend, depended[0])
    print('Анализпарной регрессии между длиной лепестка и длиной чашелистика')
    print('coef determination', r_sq)
    print('b0 coef:',model.intercept_)
    print('k coef:', model.coef_, end='\n******************\n')
    plt.scatter(iris_pd['petal length (cm)'], iris_pd['sepal length (cm)'])
    plt.plot(iris_pd['petal length (cm)'], model.predict(undepend), color='Red')
    plt.title('petal length(x) - sepal length')
    plt.show()
    prediction = model.predict(undepend)
    print(f'MAE = {(1 / 150) * sum([abs(iris_pd["petal length (cm)"][i] - prediction[i]) for i in range(len(prediction))])}')


def par_regr_petlen_sepwid():
    model = LinearRegression().fit(undepend, depended[1])
    r_sq = model.score(undepend, depended[1])
    print('Анализпарной регрессии между длиной лепестка и шириной чашелистика')
    print('coef determination', r_sq)
    print('b0 coef:',model.intercept_)
    print('k coef:', model.coef_, end='\n******************\n')
    plt.scatter(iris_pd['petal length (cm)'], iris_pd['sepal width (cm)'])
    plt.plot(iris_pd['petal length (cm)'], model.predict(undepend), color='Red')
    plt.title('petal length(x) - sepal width')
    plt.show()
    prediction = model.predict(undepend)
    print(f'MAE = {(1 / 150) * sum([abs(iris_pd["petal length (cm)"][i] - prediction[i]) for i in range(len(prediction))])}')


def par_regr_petlen_petwid():
    model = LinearRegression().fit(undepend, depended[2])
    r_sq = model.score(undepend, depended[2])
    print('Анализпарной регрессии между длиной лепестка и шириной лепестка')
    print('coef determination', r_sq)
    print('b0 coef:',model.intercept_)
    print('k coef:', model.coef_, end='\n******************\n')
    plt.scatter(iris_pd['petal length (cm)'], iris_pd['petal width (cm)'])
    plt.plot(iris_pd['petal length (cm)'], model.predict(undepend), color='Red')
    plt.title('petal length(x) - petal width')
    plt.show()
    prediction = model.predict(undepend)
    print(f'MAE = {(1/150)*sum([abs(iris_pd["petal length (cm)"][i] - prediction[i]) for i in range(len(prediction))])}')


def ost_regr_petlen_seplen():
    model = LinearRegression().fit(undepend, depended[0])
    predskazanie = model.predict(undepend)
    ls = [float(depended[0][i] - predskazanie[i]) for i in range(len(predskazanie))]
    fig = plt.figure(figsize=(12,10))
    fig = fig.add_subplot()
    fig.grid()
    plt.scatter(undepend,ls)
    plt.plot(undepend, [0]*150, color='Red')
    plt.title('Остатки от длины чашелистика')
    pg.qqplot(ls)
    plt.show()


def ost_regr_petlen_petwid():
    model = LinearRegression().fit(undepend, depended[2])
    predskazanie = model.predict(undepend)
    ls = [float(depended[2][i] - predskazanie[i]) for i in range(len(predskazanie))]
    fig = plt.figure(figsize=(12, 10))
    fig = fig.add_subplot()
    fig.grid()
    plt.scatter(undepend, ls)
    plt.plot(undepend, [0] * 150, color='Red')
    plt.title('Остатки от ширины лепестка')
    pg.qqplot(ls)
    plt.show()


def ost_regr_petlen_sepwid():
    model = LinearRegression().fit(undepend, depended[1])
    predskazanie = model.predict(undepend)
    ls = [float(depended[1][i] - predskazanie[i]) for i in range(len(predskazanie))]
    fig = plt.figure(figsize=(8, 6))
    fig = fig.add_subplot()
    fig.grid()
    plt.scatter(undepend, ls)
    plt.plot(undepend, [0] * 150, color='Red')
    plt.title('Остатки от ширины чашелистика')
    pg.qqplot(ls)
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


#парная регрессия рабочая
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
    plt.scatter(iris_pd[vhod], iris_pd[vihod])
    plt.plot(X, model.predict(X), color='Red', label='Regression line')
    plt.axhline(y=0, color='orange', linestyle='--', linewidth=1)
    plt.scatter(y, df_ostatok, color='green', label='Остатки')
    plt.xlim(0, 7)
    plt.ylim(-2, 7.5)
    plt.legend()
    plt.grid()
    plt.xlabel(vhod)
    plt.ylabel(vihod)
    plt.show()

def model_check():
    X = iris_pd.drop(columns=['petal length (cm)', 'sepal width (cm)'])
    y = iris_pd['petal length (cm)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Тренировочная  модель
    model = LinearRegression()
    model.fit(X_train, y_train)
    coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeffs'])

    # Тестовая модель
    model_test = LinearRegression()
    model_test.fit(X_test, y_test)
    coef_t = pd.DataFrame(model.coef_, X.columns, columns=['Coeffs'])

    def train():
        print('        TRAIN')
        print(coef_df)
        print(f'intercept: {model.intercept_}')
        print(f'coef determination: {model.score(X_train, y_train)}')
        print(f'Скорректированный кф детерминации {1 - (1 - model.score(X_train, y_train))*(119 / (120 - 2 - 1))}')
        print(f'coef correlation: {model.score(X_train, y_train) ** 0.5}')
        print(f'MSE : {mean_absolute_error(y_true=y_train, y_pred=model.predict(X_train))}')
        return plt.scatter(y_train, y_train - model.predict(X_train))

    def tesst():
        print('\n        TEST')
        print(coef_t)
        print(f'intercept: {model_test.intercept_}')
        print(f'coef determination: {model.score(X_test, y_test)}')
        print(f'Скорректированный кф детерминации {1 - (1 - model.score(X_test, y_test)) * (29 / (30 - 2 - 1))}')
        print(f'coef correlation: {model.score(X_test, y_test) ** 0.5}')
        print(f'MSE : {mean_absolute_error(y_true=y_test, y_pred=model.predict(X_test))}')
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

model_check()
