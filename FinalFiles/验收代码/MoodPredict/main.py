import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


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
    list_avg = []
    list_max = []
    list_min = []
    list_std = []
    list_labels = []
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


def handleData2():
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
    return newFeatures, newLabels


if __name__ == "__main__":
    # 数据导入
    allData = importData()

    # 数据预处理
    # 清理整行属性都为空的行
    allData.dropna(how="all")
    # 本来打算填充众数，因为无缺失值，后改为删除含有空属性的行，结果不变
    allData.dropna(axis=0)
    # 剩余部分缺失值也可以采用出现最频繁的值填充
    # freq_port = allData.Embarked.dropna().mode()[0] # mode返回出现最多的数据，可能出现多个，因此返回数组,取第0个
    # dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    # 输出数据形状
    allData_row = allData.shape[0]
    allDate_col = allData.shape[1]
    print("原始数据规模：", allData.shape[0], " * ", allData.shape[1])

    # 选取所有行，和最后一列之前的所有列，后面的“：-1”表示左闭右开
    features = allData.iloc[:, :-1]
    print("初始特征数目：", features.shape[1])

    # 特征提取
    # 以每10组数据为一组提取获得的20个属性的平均值，最大值，最小值，标准差，作为新的特征向量
    newFeatures, newLabels = newFeaturesAndLabelsInit()
    newFeatures, newLabels = handleData2()

    # 输出提取特征后，新的特征维度个数
    newFeatures_row = newFeatures.shape[1]
    print("特征提取后，特征数目：", newFeatures_row)

    # 归一化处理
    normal_data = (newFeatures - newFeatures.min()) / (newFeatures.max() - newFeatures.min())

    # 特征和标签分别赋值
    X = normal_data
    y = newLabels['labels']
    # 划分训练集和预测集
    # 设置stratify = y时，我们发现每次划分后，测试集和训练集中的类标签比例同原始的样本中类标签的比例相同
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    # # PCA降维过程
    # pca = PCA(n_components = 'mle',copy = False).fit(X_train)
    # X_train_pca = pca.transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # # 输出降维后的结果
    # print(X_train_pca.shape)
    # print("各成分占总方差比例：")
    # print(pca.explained_variance_ratio_ )
    # print("各成分方差：")
    # print(pca.explained_variance_ )

    # 进行预测
    # kNN
    clf = KNeighborsClassifier(n_neighbors=16, algorithm='auto', weights='distance')
    clf.fit(X_train, y_train)
    print("kNN算法预测准确率为：", clf.score(X_test, y_test))
    # 绘制图像显示不同K值对应的预测准确率
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        # loss = -cross_val_score(clf, X, y, cv=10, scoring='mean_squared_error') # for regression
        # 当cv参数是一个整型时，cross_val_score默认使用KFold 或StratifiedKFold的方法
        scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')  # for classification
        k_scores.append(scores.mean())
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    # SVC(kernel='linear')
    clf = svm.SVC(kernel='linear',decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    print("SVM算法(kernel='linear')预测准确率为：", clf.score(X_test, y_test))

    # SVC(kernel='poly')
    clf = svm.SVC(kernel='poly',decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    print("SVM算法(kernel='poly')预测准确率为：", clf.score(X_test, y_test))

    # SVC(kernel='rbf')
    clf = svm.SVC(kernel='rbf',decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    print("SVM算法(kernel='rbf')预测准确率为：", clf.score(X_test, y_test))

