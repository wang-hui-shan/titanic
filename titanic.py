import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#all_data = pd.read_csv('train.csv')
#all_data.drop(columns=['PassengerId'],inplace=True)
all_data = pd.concat([pd.read_csv('train.csv'),pd.read_csv('test.csv')])
# 新增特征
# 名字前缀
all_data['prefix'] =  all_data['Name'].apply(lambda x :  x.strip().split(',')[1].strip().split('.')[0])
# 家庭成员
all_data['Fnums'] = all_data['SibSp'] + all_data['Parch'] + 1
# Fare=0替换为np.nan
all_data['Fare'].replace(0.0,np.nan,inplace=True)

#1.Cabin列缺失值过多，暂时清除,Name,Ticket包含文本数据，暂时认为和姓名,ticket无关
all_data.drop(columns=['Cabin', 'Name', 'Ticket'],inplace=True)

# Age 用 KNNImputer 填充，Embarked 用出现最多的类别C填充
values = {'Embarked': 'C','Fare':all_data['Fare'].median(),'Age':all_data['Age'].median()}
all_data.fillna(value=values,inplace=True)

# 增加特征
# AgeBin FareBin
all_data['AgeBin'] = pd.cut(all_data['Age'].astype(int), 5)
all_data['FareBin'] = pd.cut(all_data['Fare'].astype(int), 25)

#数据处理
#将离散型值转换为数值Categorical
from sklearn.preprocessing import LabelEncoder
def labelEncoding():
    le = LabelEncoder()
    all_data['prefix']  = le.fit_transform(all_data['prefix'])
    all_data['Embarked']  = le.fit_transform(all_data['Embarked'])
    all_data['Sex']  = le.fit_transform(all_data['Sex'])
    all_data['AgeBin'] = le.fit_transform(all_data['AgeBin'])
    all_data['FareBin'] = le.fit_transform(all_data['FareBin'])
	
def oneHotEncoding():
    dum = pd.get_dummies(all_data[['Sex','Embarked','prefix','AgeBin','FareBin']],)
    data = pd.concat([all_data.drop(columns=['Sex','Embarked','prefix','AgeBin','FareBin']), dum],axis=1)
    return data

labelEncoding()
# all_data = oneHotEncoding()

X = all_data.drop(columns=['PassengerId', 'Survived'])[:891]
y = all_data['Survived'][:891]

test_X = all_data.drop(columns=['PassengerId', 'Survived'])[891:]

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
test_X = scaler.transform(test_X)

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
cv_results = cross_validate(SC , X, y, cv=10,return_estimator=True)
print("SC:",cv_results['test_score'].mean())

VC = VotingClassifier(estimators=estimators)
cv_results = cross_validate(VC , X, y, cv=10,return_estimator=True)
print("VC:",cv_results['test_score'].mean())
# 测试集预测
best_clf = cv_results['estimator'][cv_results['test_score'].argmax()]
test_y = best_clf.predict(test_X)
predict = pd.DataFrame()
predict['PassengerId'] = all_data['PassengerId'][891:]
predict['Survived'] = test_y.astype(int)
predict.to_csv('predic.csv',index=False)
"""
LR: 0.7979900124843946
RC: 0.7968539325842696
SGD: 0.7384769038701624
GNB: 0.7833832709113608
KNN: 0.8170911360799001
DTC: 0.7778027465667916
LSVC: 0.7935081148564295
SVC: 0.8294132334581773
ABC: 0.8070037453183521
BC: 0.813732833957553
GBC: 0.8395505617977529
RFC: 0.8036828963795255
SC: 0.8283021223470662
VC: 0.832796504369538
"""
