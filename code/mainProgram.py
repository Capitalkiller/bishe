# 总文件

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def importData():
    data1 = pd.read_table('data1.txt', sep=',')
    data2 = pd.read_table('data2.txt', sep=',')
    # 数据收集完成后，发现GPS经纬度基本没有变化（大小变化约在8位以后），所以删除经纬度，
    # 因收集样本有限，并且大部分人都在室内活动，情景模式基本都为震动，没有太大参考价值，删除情景模式
    data1 = data1.drop('19', 1)
    data1 = data1.drop('20', 1)
    data1 = data1.drop('23', 1)
    allData = data1.append(data2)
    return allData


def newFeaturesAndLabelsInit():
    newFeatures = pd.DataFrame(
        columns=['avg_1', 'max_1', 'min_1', 'std_1', 'avg_2', 'max_2', 'min_2', 'std_2', 'avg_3', 'max_3', 'min_3',
                 'std_3',
                 'avg_4', 'max_4', 'min_4', 'std_4', 'avg_5', 'max_5', 'min_5', 'std_5', 'avg_6', 'max_6', 'min_6',
                 'std_6',
                 'avg_7', 'max_7', 'min_7', 'std_7', 'avg_8', 'max_8', 'min_8', 'std_8', 'avg_9', 'max_9', 'min_9',
                 'std_9',
                 'avg_10', 'max_10', 'min_10', 'std_10', 'avg_11', 'max_11', 'min_11', 'std_11',
                 'avg_12', 'max_12', 'min_12', 'std_12', 'avg_13', 'max_13', 'min_13', 'std_13',
                 'avg_14', 'max_14', 'min_14', 'std_14', 'avg_15', 'max_15', 'min_15', 'std_15',
                 'avg_16', 'max_16', 'min_16', 'std_16', 'avg_17', 'max_17', 'min_17', 'std_17',
                 'avg_18', 'max_18', 'min_18', 'std_18', 'avg_21', 'max_21', 'min_21', 'std_21',
                 'avg_22', 'max_22', 'min_22', 'std_22'])
    newLabels = pd.DataFrame(columns=['labels'])
    return newFeatures, newLabels


def handleData(length, count):
    i = 0
    j = 0
    while i < (length - 10):
        j = i + 10
        data_slice = features.iloc[i:j, count]  # iloc函数传入的参数是位置索引
        list_avg.append(data_slice.mean())
        list_max.append(data_slice.max())
        list_min.append(data_slice.min())
        list_std.append(data_slice.std())
        i = j
    newFeatures['avg_' + str(count + 1)] = np.array(list_avg)
    newFeatures['max_' + str(count + 1)] = np.array(list_max)
    newFeatures['min_' + str(count + 1)] = np.array(list_min)
    newFeatures['std_' + str(count + 1)] = np.array(list_std)
    list_avg.clear()
    list_max.clear()
    list_min.clear()
    list_std.clear()


# 数据导入
allData = importData()

# 数据预处理
# 清理整行属性都为空的行
allData.dropna(how="all")

# 剩余部分缺失值采用出现最频繁的值填充
# freq_port = allData.Embarked.dropna().mode()[0] # mode返回出现最多的数据，可能出现多个，因此返回数组,取第0个
# dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# 选取所有行，和最后一列之前的所有列，后面的“：-1”表示左闭右开
features = allData.iloc[:, :-1]

# 特征提取
# 以每10组数据为一组提取获得的20个属性的平均值，最大值，最小值，标准差，作为新的特征向量
newFeatures, newLabels = newFeaturesAndLabelsInit()

list_avg = []
list_max = []
list_min = []
list_std = []
list_labels = []

handleData(features.shape[0], 0)
handleData(features.shape[0], 1)
handleData(features.shape[0], 2)
handleData(features.shape[0], 3)
handleData(features.shape[0], 4)
handleData(features.shape[0], 5)
handleData(features.shape[0], 6)
handleData(features.shape[0], 7)
handleData(features.shape[0], 8)
handleData(features.shape[0], 9)
handleData(features.shape[0], 10)
handleData(features.shape[0], 11)
handleData(features.shape[0], 12)
handleData(features.shape[0], 13)
handleData(features.shape[0], 14)
handleData(features.shape[0], 15)
handleData(features.shape[0], 16)
handleData(features.shape[0], 17)
# for count in range(17):
#     handleData(features.shape[0],count)

i = 0
j = 0
while i < (features.shape[0] - 10):
    j = i + 10
    data_slice = features.iloc[i:j, 18]
    list_avg.append(data_slice.mean())
    list_max.append(data_slice.max())
    list_min.append(data_slice.min())
    list_std.append(data_slice.std())
    i = j
newFeatures['avg_21'] = np.array(list_avg)
newFeatures['max_21'] = np.array(list_max)
newFeatures['min_21'] = np.array(list_min)
newFeatures['std_21'] = np.array(list_std)
list_avg.clear()
list_max.clear()
list_min.clear()
list_std.clear()

i = 0
j = 0
while i < (features.shape[0] - 10):
    j = i + 10
    data_slice = features.iloc[i:j, 19]
    list_avg.append(data_slice.mean())
    list_max.append(data_slice.max())
    list_min.append(data_slice.min())
    list_std.append(data_slice.std())
    list_labels.append(allData.iloc[i, -1])  # 选取allData里面第i行，最后一列作为labels
    i = j
newFeatures['avg_22'] = np.array(list_avg)
newFeatures['max_22'] = np.array(list_max)
newFeatures['min_22'] = np.array(list_min)
newFeatures['std_22'] = np.array(list_std)
newLabels['labels'] = np.array(list_labels)
list_avg.clear()
list_max.clear()
list_min.clear()
list_std.clear()
list_labels.clear()

# 归一化处理
normal_data = (newFeatures - newFeatures.min()) / (newFeatures.max() - newFeatures.min())

# 特征和标签分别赋值
X = normal_data
y = newLabels['labels']
# 划分训练集和预测集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# kNN约在70%以内，随机种子为1时65%，随机种子为0时66%
# clf = KNeighborsClassifier(n_neighbors=20)
## SVC(kernel='linear')随机种子为1时66%，随机种子为0时66%
## clf = svm.SVC(kernel='linear')
# SVC(kernel='poly')随机种子为1时57%，随机种子为0时60%
# clf = svm.SVC(kernel='poly')
# SVC(kernel='rbf')随机种子为1时64%，随机种子为0时65%
# 产生linear核函数比rbf结果更好的原因是，使用了默认参数，未对rbf参数进行调优，调优后结果好于linear
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# 将数据写入excel表格
# writer = pd.ExcelWriter('Save_Excel2.xlsx')
# data.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
# writer.save()