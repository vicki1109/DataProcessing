
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.preprocessing import  StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import time
from scipy import sparse


# In[5]:


bank = pd.read_csv('./bank/bank-full.csv', sep=';')
bank.head()


# In[6]:


bank.describe()  # 数值型


# In[14]:


bank.describe(include=['O'])


# In[15]:


bank.info()


# In[16]:


for col in bank.select_dtypes(include=['object']).columns:
    print(col + ':', bank[bank[col]=='unknown'][col].count())


# In[17]:


bank['y'].value_counts()


# In[23]:


f, ax = plt.subplots(1, 1, figsize=(4,4))
colors = ["#FA5858", "#64FE2E"]


# In[24]:


labels="no", "yes"
ax.set_title("是否认购定期存款", fontsize=16)

bank["y"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax, shadow=True, colors=colors, labels=labels, fontsize=14)

plt.axis('off')
plt.show()


# ### hist查看每个数值型特征分布情况  看出是分类还是回归

# In[25]:


bank.hist(bins=25, figsize=(14, 10))
plt.show()


# ### 2.2 类别型特征对结果的影响(受教育程度与结果)

# In[34]:


f, ax = plt.subplots(1, 1, figsize=(9,7))

palette = ["#64FE2E", "#FA5858"]

sns.barplot(x="housing", y="balance", hue="y", data=bank, palette=palette, estimator=lambda x: len(x) / len(bank) * 100)

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02), fontsize=12)
    
ax.set_xticklabels(bank["housing"].unique(), rotation=0, rotation_mode="anchor", fontsize=15)
ax.set_title("受教育程度与结果（是否认购定期存款）的关系", fontsize=20)
ax.set_xlabel("受教育程度", fontsize=15)
ax.set_ylabel("(%)", fontsize=15)

plt.show()


# ## 通过关系矩阵查看各特征之间的关系

# In[39]:


fig, ax = plt.subplots(figsize=(12, 8))
bank['y'] = LabelEncoder().fit_transform(bank['y'])
numeric_bank = bank.select_dtypes(exclude="object")
corr_numeric = numeric_bank.corr()   # 关系矩阵，以矩阵形式存储

sns.heatmap(corr_numeric, annot=True, vmax=1, vmin=-1, cmap="Blues", annot_kws={"size":15})  # 热力图，即关系矩阵
ax.set_title("Correlation Matrix", fontsize=24)
ax.tick_params(axis='y', labelsize=11.5)
ax.tick_params(axis='x', labelsize=11.5)
plt.show()


# ### 观察看出通话时长（duration）与结果(y)相关性很高。通话时长越长，则客户购买意向越高。
# ### 低于或高于平均值分为below_average和over_average两类

# In[41]:


sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.set_style('whitegrid')
avg_duration = bank['duration'].mean()

# 建立一个新特征以区分大于duration平均值和小于均值的duration
bank["duration_status"] = np.nan
lst = [bank]
for col in lst:
    col.loc[col["duration"] < avg_duration, "duration_status"] = "below_average"
    col.loc[col["duration"] > avg_duration, "duration_status"] = "above_average"
    
pct_term = pd.crosstab(bank['duration_status'], bank['y']).apply(lambda r : round(r/r.sum(), 2) * 100, axis=1)
ax = pct_term.plot(kind='bar', stacked=False, cmap='RdBu')
ax.set_xticklabels(['below_average', 'over_average'], rotation=0, rotation_mode="anchor", fontsize=18)
plt.title('The Influence of Duration', fontsize=18)
plt.xlabel('Duration Status', fontsize=18)
plt.ylabel("Percentage (%)", fontsize=18)

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()), (p.get_x(), p.get_height() * 1.02))
plt.show()
bank.drop(['duration_status'], axis=1, inplace=True)


# In[42]:


print(lst)


# In[43]:


bank.info


# In[47]:


pct_term


# In[48]:


bank['poutcome'].value_counts()


# ## 类型转换OnehotEncoder
# #### 单列LabelEncoder 、多列CategoricalEncoder

# In[50]:


## 多列文本属性列
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64, handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense'  or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)
        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or 'ignore', got %s")
            raise ValueError(template % self.handle_unknown)
        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for encoding='ordinal'")
        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]
        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1} during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))
        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])
            
            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1} during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]

            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsr()

        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# In[51]:


bank[['job', 'marital']].head(5)


# In[52]:


a = CategoricalEncoder().fit_transform(bank[['job', 'marital']])
a.toarray()


# In[53]:


a.shape


# In[54]:


bank ## 对所有数值型做标准化操作


# In[55]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# ### pipeline把多个处理节点按顺序打包在一起，逐个调用节点的fit() transform() 用最后一个节点的fit()方法来拟合数据
# 
# ### 数值型特征用StandardScaler()进行标准化，类别型特征用categoricalEncoder进行独热编码

# In[56]:


# 制作管道
# 数值型特征
numerical_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["age", "balance", "day", "campaign", "pdays", "previous", "duration"])),
    ("std_scaler", StandardScaler()),
])
# 类别特征
categorical_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["job", "education", "marital", "default", "housing", "loan", "contact", "month", "poutcome"])),
    ("cat_encoder", CategoricalEncoder(encoding='onehot-dense'))
])

# 统一管道
preprocess_pipeline = FeatureUnion(transformer_list=[
    ("numerical_pipeline", numerical_pipeline),
    ("categorical_pipeline", categorical_pipeline),
])


# ## 模型训练

# In[57]:


X = bank.drop(['y'], axis=1)
y = bank['y']
X


# In[58]:


X = preprocess_pipeline.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)


# In[59]:


# 将X转为dataframe
preprocess_bank = pd.DataFrame(X)
preprocess_bank.head(5)


# In[60]:


bank.head(5)


# ## 模型构建

# In[83]:


t_diff = []
# 逻辑回归
log_reg = LogisticRegression()
t_start = time.clock()  # 通过time记录
log_scores = cross_val_score(log_reg, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

log_reg_mean = log_scores.mean()
print(t_diff)
print(log_reg_mean)


# In[84]:


# 支持向量机
svc_clf = SVC()
t_start = time.clock()  # 通过time记录
svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

svc_mean = svc_scores.mean()
print(t_diff)
print(svc_mean)


# In[85]:


# k近邻
knn_clf = KNeighborsClassifier()
t_start = time.clock()  # 通过time记录
knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

knn_mean = knn_scores.mean()
print(t_diff)
print(knn_mean)


# In[86]:


# 决策树
tree_clf = tree.DecisionTreeClassifier()
t_start = time.clock()  # 通过time记录
tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

tree_mean = tree_scores.mean()
print(t_diff)
print(tree_mean)


# In[87]:


# 梯度提升树
grad_clf = GradientBoostingClassifier()
t_start = time.clock()  # 通过time记录
grad_scores = cross_val_score(grad_clf, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

grad_mean = grad_scores.mean()
print(t_diff)
print(grad_mean)


# In[88]:


# 随机森林
rand_clf = RandomForestClassifier()
t_start = time.clock()  # 通过time记录
rand_scores = cross_val_score(rand_clf, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

rand_mean = rand_scores.mean()
print(t_diff)
print(rand_mean)


# In[89]:


# 神经网络
neural_clf = MLPClassifier(alpha=0.01)
t_start = time.clock()  # 通过time记录
neural_scores = cross_val_score(neural_clf, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

neural_mean = neural_scores.mean()
print(t_diff)
print(neural_mean)


# In[90]:


# 朴素贝叶斯
nav_clf = GaussianNB()
t_start = time.clock()  # 通过time记录
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=3, scoring='roc_auc')
t_end = time.clock()
t_diff.append((t_end - t_start))

nav_mean = nav_scores.mean()
print(t_diff)
print(nav_mean)


# In[91]:


d = {'Classifiers':['Logistic_reg', 'SVC', 'KNN', 'Dec_Tree', 'Grad B CLF', 'Rand FC', 'Neural Classifier', 'Naives_Bayes'],
    'Crossval Mean Scores': [log_reg_mean, svc_mean, knn_mean, tree_mean, grad_mean, rand_mean, neural_mean, nav_mean],
    'time':t_diff}

result_df = pd.DataFrame(d)
result_df = result_df.sort_values(by=['Crossval Mean Scores'], ascending=False)


# ### 随机森林与梯度提升树（上百个弱分类器），训练时间相对较长，但可以并行计算，且集成模型泛化能力较强，不需要调参，效果远远好于单棵决策树；神经网络最慢，决策树最快。SVC和KNN，因为KNN要计算每个点到某点的距离，所以耗时较多，效果也一般。

# In[92]:


result_df


# ## 模型评价

# In[95]:


# 分类器AUC值与ROC曲线的参数
def get_auc(clf):
    clf = clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)
    prob = prob[:, 1]
    return roc_auc_score(y_test, prob), roc_curve(y_test, prob)
    


# In[96]:


grad_roc_scores, grad_roc_curve = get_auc(grad_clf)
neural_roc_scores,neural_roc_curve = get_auc(neural_clf)
naives_roc_scores, naives_roc_curve = get_auc(nav_clf)

grd_fpr, grd_tpr, grd_threshold = grad_roc_curve
neu_fpr, neu_tpr, neu_threshold = neural_roc_curve
nav_fpr, nav_tpr, nav_threshold = naives_roc_curve


# In[97]:


def graph_roc_curve_multiple(grd_fpr, grd_tpr, neu_fpr, neu_tpr, nav_fpr, nav_tpr):
    plt.figure(figsize=(8,6))
    plt.title('ROC Curve \n Top 3 Classifiers', fontsize=18)
    plt.plot(grd_fpr, grd_tpr, label='Gradient Boostring Classifier (Score = {:.2%})'.format(grad_roc_scores))
    plt.plot(neu_fpr, neu_tpr, label='Neural Classifier (Score = {:.2%})'.format(neural_roc_scores))
    plt.plot(nav_fpr, nav_tpr, label='Naives Bayes Classifier (Score = {:.2%})'.format(naives_roc_scores))
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5,0.5), xytext=(0.6,0.3))
    plt.legend()
    
graph_roc_curve_multiple(grd_fpr, grd_tpr, neu_fpr, neu_tpr, nav_fpr, nav_tpr)
plt.show()


# In[ ]:




