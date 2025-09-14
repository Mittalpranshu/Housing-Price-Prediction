import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
#1 LOAD DATA
housing=pd.read_csv("housing.csv")

#2 CREATE A STRATIFIED TEST SET
housing['income_cat']=pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['income_cat']):
    train_set=housing.loc[train_index].drop('income_cat',axis=1)
    test_set=housing.loc[test_index].drop('income_cat',axis=1)

# WORKING ON COPY
housing=train_set.copy()

#3.SEPERATE FEATURES AND LABELS
housing_labels=housing['median_house_value'].copy()
housing=housing.drop('median_house_value',axis=1)

#print(housing,housing_labels)

#4Seperate Numerical and Categorical Data
num_attributes=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribute=['ocean_proximity']

#5 pipelines
#   FOR NUMERIC
num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('Scaler',StandardScaler())
])
# For CAtegoric
cat_pipeline=Pipeline([
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
#FULL PIPELINE
full_pipeline=ColumnTransformer([
    ('numeric',num_pipeline,num_attributes),
    ('categorical',cat_pipeline,cat_attribute)
])
#6 TRANSFORMING DATA
housing_prepared=full_pipeline.fit_transform(housing)
print(housing_prepared.shape)
#7 Training Model
#7.1 Linear_Regession
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred=lin_reg.predict(housing_prepared)
lin_rmse=-cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
# lin_rmse=root_mean_squared_error(housing_labels,lin_pred)
print(pd.Series(lin_rmse).describe())
# print(f"The root mean squared error for Linear regression is {lin_rmse}")

#7.2 Decision_Tree
dec_tree=DecisionTreeRegressor()
dec_tree.fit(housing_prepared,housing_labels)
dec_tree_pred=dec_tree.predict(housing_prepared)
dec_rmse=-cross_val_score(dec_tree,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
# dec_rmse=root_mean_squared_error(housing_labels,dec_tree_pred)
print(pd.Series(dec_rmse).describe())
# print(f"The root mean squared error for Decison Tree is {dec_rmse}")

#7.3 RandomForest
randfor=RandomForestRegressor()
randfor.fit(housing_prepared,housing_labels)
randfor_pred=randfor.predict(housing_prepared)
# rand_rmse=root_mean_squared_error(housing_labels,randfor_pred)
rand_rmse=-cross_val_score(randfor,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
# print(f"The root mean squared error for Random Forest is {rand_rmse}")
print(pd.Series(rand_rmse).describe())























