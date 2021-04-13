import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#train_data = pd.read_csv('train.csv')
#train_data.drop(columns=['PassengerId'],inplace=True)
train_data = pd.concat([pd.read_csv('train.csv'),pd.read_csv('test.csv')])
# 新增特征
# 名字前缀
train_data['prefix'] =  train_data['Name'].apply(lambda x :  x.strip().split(',')[1].strip().split('.')[0])
# 家庭成员
train_data['Fnums'] = train_data['SibSp'] + train_data['Parch'] + 1
# Fare=0替换为np.nan
train_data['Fare'].replace(0.0,np.nan,inplace=True)

#1.Cabin列缺失值过多，暂时清除,Name,Ticket包含文本数据，暂时认为和姓名,ticket无关
train_data.drop(columns=['Cabin', 'Name', 'Ticket'],inplace=True)

#查看存在缺失的位置 train_data[train_data.isnull().values==True]
# Age Nan 暂时用 median 填充，Embarked 用第四种类别 A 替代 Nan
# values = {'Age': train_data['Age'].median(), 'Embarked': 'C'}
values = {'Embarked': 'C','Fare':train_data['Fare'].median()}
train_data.fillna(value=values,inplace=True)

from sklearn.impute import KNNImputer
imputer = KNNImputer()
train_data[['Age']] = imputer.fit_transform(train_data[['Age']])

# 增加特征
# AgeBin FareBin
train_data['AgeBin'] = pd.cut(train_data['Age'].astype(int), 5)
train_data['FareBin'] = pd.cut(train_data['Fare'].astype(int), 25)


#数据处理
#将离散型值转换为数值Categorical
from sklearn.preprocessing import LabelEncoder
def labelEncoding():
    le = LabelEncoder()
    train_data['prefix']  = le.fit_transform(train_data['prefix'])
    train_data['Embarked']  = le.fit_transform(train_data['Embarked'])
    train_data['Sex']  = le.fit_transform(train_data['Sex'])
    train_data['AgeBin'] = le.fit_transform(train_data['AgeBin'])
    train_data['FareBin'] = le.fit_transform(train_data['FareBin'])
	
def oneHotEncoding():
    dum = pd.get_dummies(train_data[['Sex','Embarked','prefix','AgeBin','FareBin']],)
    data = pd.concat([train_data.drop(columns=['Sex','Embarked','prefix','AgeBin','FareBin']), dum],axis=1)
    return data

labelEncoding()
# train_data = oneHotEncoding()

X=train_data.drop(columns=['PassengerId', 'Survived'])[:891]
y=train_data['Survived'][:891]

test_X = train_data.drop(columns=['PassengerId', 'Survived'])[891:]

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.fit_transform(test_X)

# cross_validate交叉验证
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_validate

# 模型构建

# 线性模型
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
LR = LogisticRegression()
cv_results = cross_validate(LR, X, y, cv=10)
print("LR:",cv_results['test_score'].mean())

RC = RidgeClassifier()
cv_results = cross_validate(RC, X, y, cv=10)
print("RC:",cv_results['test_score'].mean())

SGD = SGDClassifier()
cv_results = cross_validate(SGD, X, y, cv=10)
print("SGD:",cv_results['test_score'].mean())

# 贝叶斯分类器 KNN 决策树
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

GNB = GaussianNB()
cv_results = cross_validate(GNB, X, y, cv=10)
print("GNB:",cv_results['test_score'].mean())

KNN = KNeighborsClassifier()
cv_results = cross_validate(KNN, X, y, cv=10)
print("KNN:",cv_results['test_score'].mean())

DTC = DecisionTreeClassifier()
cv_results = cross_validate(DTC, X, y, cv=10)
print("DTC:",cv_results['test_score'].mean())

# SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

LSVC = LinearSVC()
cv_results = cross_validate(LSVC , X, y, cv=10)
print("LSVC:",cv_results['test_score'].mean())

SVC = SVC()
cv_results = cross_validate(SVC , X, y, cv=10)
print("SVC:",cv_results['test_score'].mean())

# 集成模型
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier

ABC = AdaBoostClassifier()
cv_results = cross_validate(ABC , X, y, cv=10)
print("ABC:",cv_results['test_score'].mean())

BC = BaggingClassifier()
cv_results = cross_validate(BC , X, y, cv=10)
print("BC:",cv_results['test_score'].mean())

GBC = GradientBoostingClassifier()
cv_results = cross_validate(GBC , X, y, cv=10)
print("GBC:",cv_results['test_score'].mean())

RFC = RandomForestClassifier()
cv_results = cross_validate(RFC , X, y, cv=10)
print("RFC:",cv_results['test_score'].mean())

# 集成模型
from sklearn.ensemble import StackingClassifier,VotingClassifier
estimators = [
    ('RFC', RFC),
    ('LR', LR),
    ("KNN",KNN),
    ("SVC",SVC)
]

SC = StackingClassifier(estimators=estimators, final_estimator=SVC)
cv_results = cross_validate(SC , X, y, cv=10)
print("SC:",cv_results['test_score'].mean())

VC = VotingClassifier(estimators=estimators)
cv_results = cross_validate(VC , X, y, cv=10)
print("VC:",cv_results['test_score'].mean())

# 测试集预测
clf = StackingClassifier(estimators=estimators, final_estimator=SVC).fit(X,y)
test_y = clf.predict(test_X)
predict = pd.DataFrame()
predict['PassengerId'] = train_data['PassengerId'][891:]
predict['Survived'] = test_y.astype(int)
predict.to_csv('predic.csv',index=False)

# 测试集预测
clf = StackingClassifier(estimators=estimators, final_estimator=SVC).fit(X,y)
test_y = clf.predict(test_X)
predict = pd.DataFrame()
predict['PassengerId'] = train_data['PassengerId'][891:]
predict['Survived'] = test_y.astype(int)
predict.to_csv('predic.csv',index=False)
"""
LR: 0.7957428214731583
RC: 0.7957428214731584
SGD: 0.7553058676654183
GNB: 0.7833832709113608
KNN: 0.8193508114856428
DTC: 0.7733707865168539
LSVC: 0.7912609238451934
SVC: 0.8305368289637952
ABC: 0.8103745318352059
BC: 0.8047191011235956
GBC: 0.8339450686641697
RFC: 0.8115230961298379
SC: 0.8283021223470662
VC: 0.8283021223470662
"""
