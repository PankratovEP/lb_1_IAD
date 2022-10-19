import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sb
import pingouin as pg
from scipy.stats import kstest


iris = load_iris()


# датафрейм имеет вид таблицы со столбами: 'sepal length (cm)' , 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])


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
threed_scatter()
