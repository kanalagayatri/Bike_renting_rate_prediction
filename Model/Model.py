# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:21:46 2018

@author: Gayatri.k
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification

df = pd.read_csv("C:\\Users\\gayatri.k\\Downloads\\bikes_data\\data\\Created_data.csv")
df.isnull().sum() 
df.describe()
sns.boxplot(data=df[['No_of_trips_started', 'No_of_trips_ended','Net_rateforday',
       'Mean_temp', 'Mean_humidity', 'Mean_windspeed', 'Subsciber', 'Customer', 'Mean_VisibilityMiles']])
fig=plt.gcf()
fig.set_size_inches(10,10)

df.Mean_temp.unique()
fig,axes=plt.subplots(2,2)
axes[0,0].hist(x="Net_rateforday",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[0,0].set_title("Variation of No_of_trips")
axes[0,1].hist(x="Mean_temp",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[0,1].set_title("Variation of Mean_temp")
axes[1,0].hist(x="Mean_humidity",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[1,0].set_title("Variation of Mean_humidity")
axes[1,1].hist(x="Mean_windspeed",data=df,edgecolor="black",linewidth=2,color='#ff4125')
axes[1,1].set_title("Variation of Mean_windspeed")
fig.set_size_inches(10,10)

cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(100,100)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.Date)]
df["month"] = [t.month for t in pd.DatetimeIndex(df.Date)]
df['year'] = [t.year for t in pd.DatetimeIndex(df.Date)]
df.head(5)
sns.factorplot(x="month",y="Net_rateforday",data=df,kind='bar',size=5,aspect=1.5)
sns.factorplot(x="year",y="Net_rateforday",data=df,kind='bar',size=5,aspect=1.5)
sns.factorplot(x="day",y="Net_rateforday",data=df,kind='bar',size=5,aspect=1.5)


new_df=df.copy()
new_df.Mean_temp.describe()
new_df['temp_bin']=np.floor(new_df['Mean_temp'])//5
new_df['temp_bin'].unique()
# now we can visualize as follows
sns.factorplot(x="temp_bin",y="Net_rateforday",data=new_df,kind='bar')

df.columns.to_series().groupby(df.dtypes).groups
df.drop('Date',axis=1,inplace=True)
df.head()                    
x_train,x_test,y_train,y_test=train_test_split(df.drop('Net_rateforday',axis=1),df['Net_rateforday'],test_size=0.25,random_state=42)


models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]
model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']
rmsle=[]
d={}
for model in range (len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    test_pred=clf.predict(x_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
d={'Modelling Algo':model_names,'RMSLE':rmsle}
rmsle_frame=pd.DataFrame(d)
rmsle_frame

