

## 											基于SVC的股票预测

一、实验目的

​		掌握支持向量机用于分类的基本原理，并应用于股票预测中。



二、实验原理

​		支持向量机（SVM）是一种线性分类器，它既可以用于回归也可以用于分类。当用于分类时也称为支持向量机用于分类（SVC）。

​		1、首先进行数据预处理，从factors和quotation中提取有用信息作为”特征“，并利用这些”特征“构建训练集和测试集。当收盘价高于开盘价，标记为 1，记为股价上涨，否则标记为0，由此得到分类所需的标签。

​		2、利用SVC对数据进行分类，本实验用了两个模型。模型一使用网格搜索（GridSearchCV）的方法进行分类，使用C 和 gamma 的参数集对分类进行控制，另外通过混淆矩阵观察分类结果。

​			模型二使用PCA+GridSearchCV进行分类，PCA是一种提取数据主要成分的数据降维方法，能减小数据规模，提高分类效率。



三、实验步骤  

​		1、标签的生成方法是由收盘价和开盘价的对比产生的，同时每天的涨跌都是明天的训练数据。

```python
#标记涨跌分类，收盘价高于开盘价，标记为 1，记为股价上涨；
#收盘价低于开盘价，标记为-1，记为股价下跌
data['label'] = np.where(data['closePrice'] > data['openPrice'], 1, -1)
data

#进行平移操作,将第二天的涨跌移回前一天作为预测值的训练集
data['label'] = data['label'].shift(-1)
data
```

​		

​		2、热力图是展示一组变量的相关系数矩阵，能非常直观地看出参数的相对大小，是常用的数据可视化方法。

```python
#各因子的相关系数
X_train_matrix = X_train.corr()

#绘制热力图
sns.set()
f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(X_train_matrix, annot=False, square=True, cmap="Reds",
linewidths=.5, vmax=1, ax=ax)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('PA_matrix_1Y_timing.png')
plt.show()
```

​		

​		3、模型一，交叉验证将数据集一部分作为训练集，一部分作为测试集，如k折交叉验证将k-1份数据作为训练集，剩下的一份用于测试。而网格搜索就是一种基于交叉验证的自动调参方法，它用交叉验证遍历所有参数然后选出其中的最优参数。

​			C参数是惩罚系数，代表对误差的容忍程度，C越大对误差容忍度越低。gramma参数隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少。

​			使用网格搜索可选出最优的C参数和gramma参数以便得到最理想的训练结果。

```matlab
#使用网格搜索/交叉验证，GridSearchCV/ Crossvalidation，使用 C 和 gamma 的参数集
#构造字典-参数集 1
param_1 = [
{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10 ,100, 1000], 'kernel':['linear']},
{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10 ,100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel':['rbf']},
]
grid_1 = GridSearchCV(SVC(), param_1, cv=5)
#训练模型
grid_1.fit(X_train_scaler, Y_train)
grid_1.best_estimator_
grid_1.scorer_
pred_3 = grid_1.predict(X_test_scaler)
score_3 = grid_1.score(X_train_scaler, Y_train)
#准确率和准确个数
accuracy_3 = metrics.accuracy_score(Y_test, pred_3)
num_3 = metrics.accuracy_score(Y_test, pred_3, normalize=False)
precision_3 = metrics.precision_score(Y_test, pred_3)
recall_3 = metrics.recall_score(Y_test, pred_3)
f1_score_3 = metrics.f1_score(Y_test, pred_3)
#构建结果
results = DataFrame(index=['score', 'accuracy', 'num of acc', 'precision', 'recall', 'f1_score'])
results['Model_1'] = [score_3, accuracy_3, num_3, precision_3, recall_3, f1_score_3]
results
```

​		

​		4、模型二，同样使用网格搜索的方法寻找最优参数，但不同的是网格搜索前使用PCA进行数据降维，PCA把主要成分当作数据集的特征，能减少计算成本，但同时结果的准确性也会受到一定影响。

```python
#在标准化的基础上进行 PCA
#设置阈值为 95%
pca = PCA(n_components=0.95)
pca.fit(X_train_scaler)
#主成分
pca.components_
pca.components_.shape
#主成分的个数
pca.n_components_
pca.explained_variance_ratio_
#使用相应的 PCA 参数对 feature 部分进行降维，首先是训练集
X_train_scaler_pca = pca.transform(X_train_scaler)
X_train_scaler_pca.shape
#使用同样的 PCA 参数对测试集进行降维
X_test_scaler_pca = pca.transform(X_test_scaler)
X_test_scaler_pca.shape
#此模型，PCA 后使用 Gridsearch CV 进行参数优化
#沿用参数集 1
#构造交叉函数
grid_2 = GridSearchCV(SVC(), param_1, cv=5)
```

​		

​		5、另外，本实验还通过混淆矩阵来判断分类性能。混淆矩阵将所有类别的预测结果与真实结果按类别放置到了同一个表里，通过混淆矩阵能直观地看出正确和错误分类的数量。

```python
#混淆矩阵
cnf_matrix_3 = confusion_matrix(Y_test, pred_3)
np.set_printoptions(precision=2)

# 绘制非标准的混淆矩阵
plt.figure()
plot_confusion_matrix(cnf_matrix_3, classes=[-1, 1], title='Confusion matrix, without normalization')
```



四、实验结果

​		1、heatmap如下，能看出各个指标的相对大小，它们直接影响着预测结果。

![1](C:\Users\magic\OneDrive\桌面\1.png)



​		2、混淆矩阵

​		左上角为TP（True Positive），即预测值和真实值都为1

​		右上角为FP（False Positive），即预测值为1，但真实值为0

​		左下角为FN（False Negative），即预测值为0，但真实值为1

​		右下角为TN（True Negative），即预测值和真实值都为0

​		我们希望的结果是TP和TN越大越好，因为这代表了正确的预测，相反第一和第三象限的值越小越好。

​		由图1图2，可见模型一FP小于模型二，TP大于模型二，效果更好。

​		图3的正确率结果也验证了这一点。

![2](C:\Users\magic\OneDrive\桌面\2.png)



![3](C:\Users\magic\OneDrive\桌面\3.png)

​									![4](C:\Users\magic\OneDrive\桌面\4.PNG)

​						

​		3、最终得到的收益率和累计收益率曲线如下。整体上两个模型的预测结果的变化趋势与实际情况还是较为接近的。但与实际对比，显然模型一的预测结果更为理想，原因是模型二使用PCA方法提取了数据集95%的特征而模型一使用了完整的数据集。

![5](C:\Users\magic\OneDrive\桌面\5.png)

![6](C:\Users\magic\OneDrive\桌面\6.png)



五、实验心得

​		本次实验将SVC应用于实用性很强股票预测中，准确率在50%左右，结果较为可观。实验中用到的多种数据处理方法以及数据可视化方法如热力图、混淆矩阵很值得学习。

​		PCA作为一种常用的数据降维方法在本实验中导致了较大误差，可见对准确性要求较高的应用应谨慎使用。