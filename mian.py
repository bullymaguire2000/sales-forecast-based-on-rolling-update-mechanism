import pandas as pd
import numpy as np
import sklearn.datasets
import copy
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import docplex.mp.model as cpx
from numpy import inf
from numpy import zeros
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor

'''数据介绍
文件名称："JD_2018_3_Sales_Dataset.csv"
数据来源：JD.com: Transaction Level  Data for the 2020 MSOM Data Driven Research Challenge
数据的简介：来源于京东2018年3月份某一种商品的销售数据，包含销售量和促销数据
属性介绍：
1.overall_discount(总促销折扣)
2.direct_discount（直接折扣）
3.quantity_discount（凑单折扣）
4.coupon_discount(优惠券折扣)
5.overall_sale（促销活动后总销售数据）
数据大小：31×5'''

#   步骤1：将原始的京东销售量数据进行转化，转化为6天数据合为1条数据的数据类型。
#   读取原始数据
order_daily = pd.read_csv('JD_2018_3_Sales_Dataset.csv')
#   对数据进行转换
new_order = []  # 创建一个空列表，存放转换后的数据
for i in range(26): # 原来是31条数据，6条数据转换为1条数据后变为26条数据
    j, k, m, n = i+1, i+2, i+3, i+4
    new_order.append(np.array(order_daily[i:i+1][:].values.tolist()
                              +order_daily[j:j+1][:].values.tolist()
                              +order_daily[k:k+1][:].values.tolist()
                              +order_daily[m:m+1][:].values.tolist()
                              +order_daily[n:n+1][:].values.tolist()
                              +order_daily[n+1:n+2][:].values.tolist()).flatten())

new_order = pd.DataFrame(new_order, columns=['overall_discount1', 'direct_discount1', 'quantity_discount1', 'coupon_discount1', 'overall_sale1',
                                            'overall_discount2', 'direct_discount2', 'quantity_discount2', 'coupon_discount2', 'overall_sale2',
                                            'overall_discount3', 'direct_discount3', 'quantity_discount3', 'coupon_discount3', 'overall_sale3',
                                            'overall_discount4', 'direct_discount4', 'quantity_discount4', 'coupon_discount4', 'overall_sale4',
                                            'overall_discount5', 'direct_discount5', 'quantity_discount5', 'coupon_discount5', 'overall_sale5',
                                            'overall_discount6', 'direct_discount6', 'quantity_discount6', 'coupon_discount6', 'overall_sale6'])
print(new_order)

#   步骤2：定义误差计算方法（MAPE)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


#   步骤3：使用模型来进行预测，并对结果进行作图，预测的方法如下介绍
'''输入x为:'overall_discount1', 'direct_discount1', 'quantity_discount1', 'coupon_discount1', 'overall_sale1',
                                            'overall_discount2', 'direct_discount2', 'quantity_discount2', 'coupon_discount2', 'overall_sale2',
                                            'overall_discount3', 'direct_discount3', 'quantity_discount3', 'coupon_discount3', 'overall_sale3',
                                            'overall_discount4', 'direct_discount4', 'quantity_discount4', 'coupon_discount4', 'overall_sale4',
                                            'overall_discount5', 'direct_discount5', 'quantity_discount5', 'coupon_discount5', 'overall_sale5',
                                            'overall_discount6', 'direct_discount6', 'quantity_discount6', 'coupon_discount6'
    输出为：'overall_sale6'
    预测的目的就是通过前5天的销售数据来预测第6天的销售量，因为在第6天开始销售前，是可以提前确定第6天的促销活动，包括各个折扣的力度多少，所以
    'overall_discount6', 'direct_discount6', 'quantity_discount6', 'coupon_discount6'是作为输入
'''
#   3.1 定义线性回归、lasso、ridge回归的类方法
#   1.Linear Regression
class linearRegression():
    def __init__(self):
        self.ws=None
        pass

    def fit(self,x,y):
        m = x.shape[0]
        X = np.concatenate((np.ones((m, 1)), x), axis=1)
        xMat = np.mat(X)
        yMat = np.mat(y.reshape(-1, 1))
        xTx = xMat.T * xMat

        ws = xTx.I * xMat.T * yMat
        #ws[0]就是b
        self.ws=ws
        return ws

    def predict(self,x_test):
        x_test=np.mat(x_test)
        y_pred=self.ws[1:,:]*x_test+self.ws[0,:]
        return y_pred

#   LassoRegression
class LassoRegression():
    def __init__(self):
        self.w=None
        pass
    def fit(self,x,y,epochs,lr,lambda0):
        m = x.shape[0]
        X=np.concatenate((np.ones((m,1)),x),axis=1)
        x_mat=np.mat(X)
        y_mat=np.mat(y.reshape(-1,1))
        w = np.ones(X.shape[1]).reshape(-1, 1)
        out_w=copy.copy(w)
        for i, item in enumerate(w):
            for j in range(epochs):
                h = x_mat * w
                gradient = x_mat[:,i].T * (h - y_mat)/m + lambda0 * np.sign(w[i])
                w[i] = w[i] - gradient * lr
                if abs(gradient) < 1e-3:
                    break
            out_w = np.array(list(map(lambda x: abs(x) < 1e-3, out_w - w)))
            if out_w.all():
                break
        self.w=w
        return w

    def predict(self,x_test):
        x_test = np.mat(x_test)
        y_pred = self.w[1:, :]*x_test + self.w[0, :]
        return y_pred


#   RidgeRegression
class RidgeRegression():
    def __init__(self):
        self.w=None
        pass

    def fit(self,x,y,epoch,lr, lambda1):
        m = x.shape[0]
        X = np.concatenate((np.ones((m, 1)), x), axis=1)
        n = X.shape[1]
        w = np.mat(np.ones((n, 1)))
        x_mat=np.mat(X)
        y_mat=np.mat((y.reshape(-1,1)))
        for i in range(epoch):
            gradient = x_mat.T * (x_mat * w - y_mat) / m + lambda1 * w
            w=w-lr*gradient

        self.w=w
        return w

    def predict(self, x_test):
        x_test1 = np.mat(x_test)
        y_pred = self.w[1:,:]*x_test1 + self.w[0, :]
        return y_pred


#   3.2 进行拟合，用前10条数据作为train集合，第11条数据作为test集合，第二次用2-11条数据作为train，第12条数据作为test
#   先选择x和y
x = new_order.drop(['overall_sale6'], axis=1)
y = new_order.loc[:, 'overall_sale6']
#   设置误差存放列表
linear_mape = []
ridge_mape = []
lasso_mape = []
for i in range(1, 15):
    x_train = np.array(x.loc[i:i+9, :])
    y_train = np.array(y.loc[i:i+9])
    x_test = np.array(x.loc[i+10:i+10, :])
    y_test = np.array(y.loc[i+10:i+10])
    '''model1 = linearRegression()
    model2 = RidgeRegression()
    model3 = LassoRegression()
    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train, 10000, 0.05, 40)   # 参数选择依据经验值
    model3.fit(x_train, y_train, 10000, 0.05, 40)
    linear_mape.append(mape(y_test, model1.predict(x_test)))
    ridge_mape.append(mape(y_test, model2.predict(x_test)))
    lasso_mape.append(mape(y_test, model3.predict(x_test)))'''  # 本来想用自己写的类来做，但是最后结果出来误差非常大，而且有的根本算不出来，所以决定用sklearn
    model1 = LinearRegression()
    model2 = Ridge()
    model3 = Lasso()
    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model3.fit(x_train, y_train)
    linear_mape.append(mape(y_test, model1.predict(x_test)))
    ridge_mape.append(mape(y_test, model2.predict(x_test)))
    lasso_mape.append(mape(y_test, model3.predict(x_test)))

print(linear_mape, ridge_mape, lasso_mape, sep='\n')





#   3.3 编写SVR函数，并用其进行拟合
opt_model = cpx.Model(name="QP Model")


def linear_kernel(x, y):
    return np.dot(x, y.T)


def poly_kernel(x, y, cef_):
    return (1 + np.dot(x, y.T)) ** cef_


class dual_svr():
    def __init__(self, x, y, c=1.2, fai=1, kernel='poly_kernel'):
        self.x = x
        self.xx = np.dot(x, x.T)
        self.y = y
        m, n = self.xx.shape
        self.set_m = range(m)
        self.set_n = range(n)
        self.c = c
        self.fai = fai

    def fit(self):
        ##添加变量
        aphli1 = {(m): opt_model.continuous_var(name="aphli1_{0}".format(m)) for m in self.set_m}
        aphli2 = {(m): opt_model.continuous_var(name="aphli2_{0}".format(m)) for m in self.set_m}
        # aphlj1 = {(n): opt_model.continuous_var(name="aphlj1_{0}".format(n)) for n in self.set_n}
        # aphlj2 = {(n): opt_model.continuous_var(name="aphlj2_{0}".format(n)) for n in self.set_n}
        b = {(n): opt_model.continuous_var(name="b_{0}".format(n)) for n in self.set_n}
        songchi1 = {(m): opt_model.continuous_var(name="songchi1_{0}".format(m)) for m in self.set_m}
        songchi2 = {(m): opt_model.continuous_var(name="songchi2_{0}".format(m)) for m in self.set_m}
        ##添加约束

        opt_model.add_constraint(opt_model.sum(aphli1[m] - aphli2[m] for m in self.set_m) == 0)
        for m in self.set_m:
            opt_model.add_constraint(aphli2[m] >= 0)
        for m in self.set_m:
            opt_model.add_constraint(aphli1[m] <= self.c)
        '''
        ##kkt
        for m in self.set_m:
            for n in self.set_n:

        fx=opt_model.sum((aphli1[m]-aphli2[m])*self.xx[m] for m in self.set_m)+b1
        for m in self.set_m:
            opt_model.add_constraint(aphli1[m]*(self.y[m]-fx-self.fai-songchi1)==0)
        for m in self.set_m:
            opt_model.add_constraint(aphli2[m]*(fx-self.y[m]-self.fai-songchi2)==0)
        for m in self.set_m:
            opt_model.add_constraint(aphli1[m]*aphli2[m]==0)

        for m in self.set_m:
            opt_model.add_constraint((self.c-aphli1[m])*songchi1==0)

        for m in self.set_m:
            opt_model.add_constraint((self.c-aphli2[m])*songchi2==0)
        '''

        ##添加目标函数
        sum1 = opt_model.sum(
            self.y[m] * (aphli1[m] - aphli2[m]) - self.fai * (aphli1[m] + aphli2[m]) for m in self.set_m)
        sum2 = opt_model.sum(
            (aphli1[m] - aphli2[m]) * (aphli1[n] - aphli2[n]) * self.xx[m][n] for m in self.set_m for n in self.set_n)
        obj = sum1 - 1 / 2 * sum2
        opt_model.maximize(obj)
        opt_model.solve()
        list_aphli1 = []
        list_aphli2 = []
        w = 0
        temp2 = 0
        for i in self.set_m:
            list_aphli1.append(aphli1[i].solution_value)
            list_aphli2.append(aphli2[i].solution_value)
            w += (aphli1[i].solution_value - aphli2[i].solution_value) * self.x[i]
            temp2 += ((aphli1[i].solution_value - aphli2[i].solution_value) * self.x[i].T)
        b = y[i] + self.fai - np.dot(temp2, self.x[i])
        return w, b



# dataset = np.array([[3, 3, 5], [4, 3, 6], [1, 1, 3]])
# y = np.array([10, 11, -12])
#
#
# du_svr = dual_svr(dataset, y)
# w, b = du_svr.fit()
# print(np.dot(w, np.array([1, 2, 3]))+b)
SVR = []
for i in range(1, 15):
    x_train = np.array(x.loc[i:i+9, :])
    y_train = np.array(y.loc[i:i+9])
    x_test = np.array(x.loc[i+10:i+10, :])
    y_test = np.array(y.loc[i+10:i+10])

    model = dual_svr(x_train, y_train)
    w, b = model.fit()
    SVR.append(mape(y_test, np.dot(np.array(w), x_test.T)+b))
print(SVR)




# 3.4 编写随机森林类，并进行拟合预测
def get_Datasets(x,y):
    return np.concatenate((x, y.reshape((-1, 1))), axis=1)

def splitDataSet(x,n_folds):
    fold_size=len(x)/n_folds
    data_split=[]
    begin=0
    end=fold_size
    for i in range(n_folds):
        data_split.append(x[begin:end,:])
        begin=end
        end+=fold_size
    return data_split


def get_subsamples(x,n):
    subDataSet=[]
    for i in range(n):
        index=[]
        for k in range(len(x)):
            index.append(np.random.randint(len(x)))
        subDataSet.append(x[index,:])
    return subDataSet


def binSplitDataSet(x, feature, value):
    mat0 = x[np.nonzero(x[:, feature] > value)[0], :]
    mat1 = x[np.nonzero(x[:, feature] < value)[0], :]
    return mat0, mat1

def regErr(x):
    return np.var(x[:,-1])*np.shape(x)[0]

def regLeaf(x):
    return np.mean(x[:,-1])

def MostNumber(x):
    len0=len(np.nonzero(x[:,-1]==0)[0])
    len1=len(np.nonzero(x[:,-1]==1)[0])
    if len0>len1:
        return 0
    else:
        return 1

def gini(x):
    corr=0.0
    for i in set(x[:,-1]):
        corr+=(len(np.nonzero(x[:,-1]==i)[0])/len(x))**2
    return 1-corr


def select_best_feature(x, m, alpha="huigui"):
    f = x.shape[1]
    index = []
    bestS = inf;
    bestfeature = 0;
    bestValue = 0;
    if alpha == "huigui":
        S = regErr(x)
    else:
        S = gini(x)

    for i in range(m):
        index.append(np.random.randint(f))
        for feature in index:
            for splitVal in set(x[:, feature]):  # set() 函数创建一个无序不重复元素集，用于遍历这个特征下所有的值
                mat0, mat1 = binSplitDataSet(x, feature, splitVal)
                if alpha == "huigui":
                    newS = regErr(mat0) + regErr(mat1)  # 计算每个分支的回归方差
                else:
                    newS = gini(mat0) + gini(mat1)  # 计算被分错率
                if bestS > newS:
                    bestfeature = feature
                    bestValue = splitVal
                    bestS = newS
        if (S - bestS) < 0.001 and alpha == "huigui":
            return None, regLeaf(x)
        elif (S - bestS) < 0.001:

            return None, MostNumber(x)
        return bestfeature, bestValue


def createTree(x,alpha="huigui",m=20,max_level=10):
    bestfeature,bestValue=select_best_feature(x,m,alpha=alpha)
    if bestfeature==None:
        return bestValue
    retTree={}
    max_level-=1
    if max_level<0:
        return regLeaf(x)
    retTree['bestFeature']=bestfeature
    retTree['bestVal']=bestValue
    lSet,rSet=binSplitDataSet(x,bestfeature,bestValue)
    retTree['right']=createTree(rSet,alpha,m,max_level)
    retTree['left']=createTree(lSet,alpha,m,max_level)
    return retTree

def RondomForest(x,n,alpha="huigui"):
    Trees=[]
    for i in range(n):
        #X_train, X_test, y_train, y_test = train_test_split(x[:,:-1], x[:,-1], test_size=0.33, random_state=42)
        #X_train=np.concatenate((X_train,y_train.reshape((-1,1))),axis=1)
        Trees.append(createTree(x,alpha=alpha))
    return Trees


def treeForecast(trees, x, alpha="huigui"):
    if alpha == "huigui":
        if not isinstance(trees, dict):
            return float(trees)

        if x[trees['bestFeature']] > trees['bestVal']:
            if type(trees['left']) == 'float':
                return trees['left']
            else:
                return treeForecast(trees['left'], x, alpha)
        else:
            if type(trees['right']) == 'float':
                return trees['right']
            else:
                return treeForecast(trees['right'], x, alpha)
    else:
        if not isinstance(trees, dict):
            return int(trees)

        if x[trees['bestFeature']] > trees['bestVal']:
            if type(trees['left']) == 'int':
                return trees['left']
            else:
                return treeForecast(trees['left'], x, alpha)
        else:
            if type(trees['right']) == 'int':
                return trees['right']
            else:
                return treeForecast(trees['right'], x, alpha)


def createForeCast(trees,test,alpha="huigui"):
    cm=len(test)
    yhat=np.mat(zeros((cm,1)))
    for i in range(cm):                                     #
        yhat[i,0]=treeForecast(trees,test[i,:],alpha)    #
    return yhat



def predictTree(Trees,test,alpha="huigui"):
    cm=len(test)
    yhat=np.mat(zeros((cm,1)))
    for trees in Trees:
        yhat+=createForeCast(trees,test,alpha)
    if alpha=="huigui": yhat/=len(Trees)
    else:
        for i in range(len(yhat)):

            if yhat[i,0]>len(Trees)/2:
                yhat[i,0]=1
            else:
                yhat[i,0]=0
    return yhat

def getall_data(data1,data2):
    return np.concatenate((data1,data2),axis=0)


x_train = np.array([[8,9,10,11,12,13,15], [11,12,13,14,16,17,18], [15,16,17,18,19,20,21]])
y_train = np.array([20, 33, 42])
x_test = np.array([[88,96,105,119,122,134,159]])
y_test = np.array([200])
dataSet=get_Datasets(x_train,y_train)
dataSet1=get_Datasets(x_test,y_test)
data=getall_data(dataSet,dataSet1)
RomdomTrees=RondomForest(data,4,alpha="huigui")
test=data
yhat = predictTree(RomdomTrees, test, alpha="huigui")
print(yhat.T)
print(data[-1,-1].T - yhat[-1])

'''在算完之后发现随机森林误差非常大，跟我们预测中的误差不符合，因为集成学习对付这种数据应该非常厉害，那可能我们类某些地方写错，但是因为实在不会改了，就用了sklearn中ensemble来拟合
#   存放随机森林的误差
RandomForests = []
for i in range(1, 15):
    x_train = np.array(x.loc[i:i+9, :])
    y_train = np.array(y.loc[i:i+9])
    x_test = np.array(x.loc[i+10:i+10, :])
    y_test = np.array(y.loc[i+10:i+10])
    dataSet = get_Datasets(x_train, y_train)
    dataSet1 = get_Datasets(x_test, y_test)
    data = getall_data(dataSet, dataSet1)
    RomdomTrees = RondomForest(data, 4, alpha="huigui")
    test = data
    yhat = predictTree(RomdomTrees, test, alpha="huigui")
    RandomForests.append(mape(y_test, data[-1, -1].T - yhat[-1]))
'''
RandomForests = []
for i in range(1, 15):
    x_train = np.array(x.loc[i:i+9, :])
    y_train = np.array(y.loc[i:i+9])
    x_test = np.array(x.loc[i+10:i+10, :])
    y_test = np.array(y.loc[i+10:i+10])
    clf = RandomForestRegressor(bootstrap=True,
                             max_depth=10,
                             max_features='auto',
                             min_samples_leaf=1,
                             min_samples_split=2,
                             n_estimators=20)
    clf.fit(x_train, y_train)
    RandomForests.append(mape(y_test, clf.predict(x_test)))

# 3.5 对不同模型的误差分布作图,最后发现竟然是lasso表现最好！！！！！！而不是随机森林！！！！！但是本次作业中都没进行参数选择，所以得到的结果具有一定缺陷
MAPE_3_linearmodels = np.array([linear_mape, ridge_mape, lasso_mape, SVR, RandomForests])
ax = plt.subplot()
ax.boxplot(MAPE_3_linearmodels.T) # 绘图
ax.set_xlabel('模型名称')
ax.set_xticklabels(['linear', 'ridge', 'lasso', 'svr', 'random forests'], fontsize=10)
ax.set_ylabel('MAPE%')
ax.set_title('不同模型的MAPE分布图')
plt.show()