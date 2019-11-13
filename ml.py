import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

print(iris.keys())
print('--------------------------------------------------------------------------------')
print(iris.DESCR)
print('--------------------------------------------------------------------------------')
print(iris.data)
print('--------------------------------------------------------------------------------')
print(iris.data.shape)
print('--------------------------------------------------------------------------------')
print(iris.feature_names)
print('--------------------------------------------------------------------------------')
print(iris.target)
print('--------------------------------------------------------------------------------')
print(iris.target.shape)
print('--------------------------------------------------------------------------------')
print(iris.target_names)
print('--------------------------------------------------------------------------------')
X = iris.data[:, :2]  # 通常用X表示样本集
print(X.shape)
print('--------------------------------------------------------------------------------')
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
print('--------------------------------------------------------------------------------')
y = iris.target
X[y == 0, 0]
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='*')  # 筛选出标签为0的所有样本的length和width的坐标
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='o')  # 筛选出标签为0的所有样本的length和width的坐标
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='blue', marker='x')  # 筛选出标签为0的所有样本的length和width的坐标
plt.show()

# 机器学习入门算法 -- KNN
# 1.思想非常简单好理解
# 2.应用了很简单的数学知识
# 3.便于理解机器学习算法过程中的细节问题
# 4.完整刻画了机器学习的流程

raw_data_X = [[3.3953312, 2.331732],
              [3.11007384, 1.7815396],
              [1.3438331, 3.3645234],
              [3.5814887, 4.6796554],
              [2.2854634, 2.8647631],
              [7.4214861, 4.6931481],
              [5.7463214, 3.5314487],
              [9.1756874, 2.5168973],
              [7.7923146, 3.4256789],
              [7.9365481, 0.7951612]]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)
# plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
# plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='r')
# plt.show()

x = np.array([8.0936, 3.3673])  # 带预测分类样本
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='r')
plt.scatter(x[0], x[1], color='b')
plt.show()

# KNN计算待预测的样本点和已知所有样本点的距离，然后对距离进行排序，进而得出所属的类别
from math import sqrt

distance = [sqrt(np.sum(x_train - x) ** 2) for x_train in X_train]
print(distance)
# 计算好的距离排序，采用下述方法进行排序后返回值是索引位置
nearest = np.argsort(distance)
print(nearest)
k = 6
topK_y = [y_train[i] for i in nearest[0:k]]
print(topK_y)
# 统计前k个距离中各个值的频次
from collections import Counter

votes = Counter(topK_y)
print(votes)
# 取票数最多的一个元素信息
print(votes.most_common(1))
predict_y = votes.most_common(1)[0][0]
print(predict_y)

# KNN 算法的缺点
# 1.效率低
# 2.高度数据相关
# 3.结果没有可解释性
# 4.维度灾难


# 自己动手使用sklearn 库中KNeighborsClassifier 实现上述过程
