# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:54:56 2019

@author: vkovvuru
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')

meal_info=pd.read_csv('meal_info.csv')

center_info=pd.read_csv('fulfilment_center_info.csv')

df1=pd.merge(train,meal_info, on='meal_id')

df=pd.merge(df1,center_info,on='center_id')

label_encode_columns = [
                        'center_type', 
                        'category', 
                        'cuisine']

le = LabelEncoder()

for col in label_encode_columns:
    le.fit(df[col])
    df[col + '_encoded'] = le.transform(df[col])
    
df_train=df.drop(['center_type','category','cuisine'],axis=1)
    
x=df_train.drop(['num_orders'],axis=1)  
  
y=df_train['num_orders']

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.15, random_state=123)

rf = RandomForestRegressor(n_estimators=200)

# fit rf_model_on_full_data on all data from the 
rf.fit(X_train, y_train)

pred=rf.predict(X_test)
log_error=mean_squared_log_error(y_test, pred)
print("Mean_Squared_Log_error: ", log_error )


df1_test=pd.merge(test,meal_info, on='meal_id')

df_test=pd.merge(df1_test,center_info,on='center_id')    
    
      
label_encode_columns = [
                        'center_type', 
                        'category', 
                        'cuisine']

le = LabelEncoder()

for col in label_encode_columns:
    le.fit(df_test[col])
    df_test[col + '_encoded'] = le.transform(df_test[col])

   
df_test1=df_test.drop(['center_type','category','cuisine'],axis=1)
    
out=rf.predict(df_test1)   
    
output = pd.DataFrame({'id': df_test1.id,
                       'num_orders': out})

output.to_csv('submission_rf.csv', index=False)    
    
################################################################################

df_train.to_csv('DF.csv',index=False)

df_test2=df_test1
df_test2['num_orders']=out

df_test2.to_csv('predicted_sales.csv',index=False)

pd.concat([df_train,df_test2],sort=False).to_csv('Total_data.csv',index=False)



################################################################################

'''Actual_sales'''
a1=df_train.groupby(['week']).mean()
#aaa=pd.DataFrame(a1)
#aaa.to_csv('aaa.csv',index=False)

'''Actual_Data_prediction'''
pred1=rf.predict(x)
g2=pd.DataFrame({'id':df.index,'week':df.week,'num_orders':pred1})
a2=g2.groupby('week').mean()
#a2.to_csv('Predicted_Sales.csv',index=False)

'''Future_Sales'''
g=pd.DataFrame({'id':test.id,'week':test.week,'num_orders':out})
g1=pd.DataFrame({'week':test.week,'num_orders':out})
a=g1.groupby('week').mean()
#a.to_csv('Future_sales.csv',index=False)


''' Graph dispaly'''
fig = plt.figure()
plt.plot(a1.index,a1['num_orders'],'-o',label="Actual_Sales")
plt.plot(a2.index,a2['num_orders'],'-o',label="Predicted_Sales")
plt.plot(a.index,a['num_orders'],'-o',label="Future_Sales")
fig.suptitle('Sales_Data', fontsize=20)
plt.xlabel('Week', fontsize=18)
plt.ylabel('No.Of Orders', fontsize=16)
plt.legend()
plt.show



df_test2.plot.scatter('week','num_orders',c='meal_id',colormap='viridis')
plt.tight_layout()
plt.show()










