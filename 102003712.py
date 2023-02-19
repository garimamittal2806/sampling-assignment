import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


data = pd.read_csv("Creditcard_data.csv")
# print(data)

x = data.drop("Class", axis='columns')
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

lr1 = LogisticRegression()
lr1.fit(X_train_res, y_train_res.ravel())
predictions = lr1.predict(X_test)


aux1 = X_train_res
aux1['class'] = y_train_res
df = aux1

# simple random sampling
sampled_df = df.sample(frac=0.7, random_state=42, replace=True)
# print(sampled_df)

# simple random sampling without replacement
samp_rep_df = df.sample(frac=0.7, random_state=42, replace=False)

# convenience sampling

convenience_sample = pd.concat([df.head(380), df.tail(380)])

# stratified sampling
n = int((1.96*1.96 * 0.5*0.5)/((0.05)**2))

strata = df.groupby('class')
stratified_df = strata.apply(lambda x: x.sample(n))


# cluster sampling

kmeans = KMeans(n_clusters=6, random_state=42).fit(df)
cluster_assignments = kmeans.labels_

selected_clusters = [1, 3, 5]

cluster_series = pd.Series(cluster_assignments)

df["cluster"] = cluster_series

df_cluster_sample = pd.DataFrame()
for i in selected_clusters:
    aux3 = df.loc[df["cluster"] == i]
    df_cluster_sample = df_cluster_sample.append(aux3, ignore_index=True)

df = df.drop('cluster', axis="columns")
df_cluster_sample = df_cluster_sample.drop('cluster', axis="columns")


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier()
}

X_train_now = sampled_df.drop("class", axis='columns')
y_train_now = sampled_df['class']
result_simple_rep = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_simple_rep.append(accuracy)

X_train_now = samp_rep_df.drop("class", axis='columns')
y_train_now = samp_rep_df['class']
result_samp_rep_df = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_samp_rep_df.append(accuracy)

# convinience sampling

X_train_now = convenience_sample.drop("class", axis='columns')
y_train_now = convenience_sample['class']
result_convenience_sample = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_convenience_sample.append(accuracy)


X_train_now = stratified_df.drop("class", axis='columns')
y_train_now = stratified_df['class']
result_stratified_df = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_stratified_df.append(accuracy)


X_train_now = df_cluster_sample.drop("class", axis='columns')
y_train_now = df_cluster_sample['class']
result_df_cluster_sample = []
for model_name, model in models.items():
    model.fit(X_train_now, y_train_now)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result_df_cluster_sample.append(accuracy)


final_result = pd.DataFrame()
final_result['simple random with replacement'] = result_simple_rep
final_result['simple random without replacement'] = result_samp_rep_df
final_result['convenience sampling'] = result_convenience_sample
final_result['stratified sampling'] = result_stratified_df
final_result['cluster sampling'] = result_df_cluster_sample

rows = {
    0: 'Logistic Regression',
    1: 'Random Forest',
    2: 'AdaBoost',
    3: 'K-Nearest Neighbors',
    4: 'XGBoost'
}

table_n = final_result.rename(index=rows)

print("final output: ")
print(table_n)
