
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings("ignore")
#Read Files:
train = pd.read_csv(r'C:\Users\Alka Anil\Desktop\ML PROJECT\train.csv')
test = pd.read_csv(r'C:\Users\Alka Anil\Desktop\ML PROJECT\test.csv')


# In[2]:


pwd #default path


# In[3]:


#DATA EXPLORATION


# In[4]:


train.describe()


# In[5]:


test.describe()


# In[6]:


#to check for duplicates
idsUnique = len(set(test.Item_Identifier))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")


# In[7]:


len(test.columns) #column length


# In[8]:


len(train.columns)


# In[9]:


len(test) #row length


# In[10]:


len(train)


# In[11]:


train.head()


# In[12]:


test.head()


# In[13]:


train.info()


# In[14]:


test.info()


# In[15]:


#Create source column to later separate the data easily
train['source']='train'
test['source']='test'


# In[16]:


#combine data to perform certain tasks together and devide it later 
data=pd.concat([train,test],ignore_index=True,sort=False)


# In[17]:


data.info()


# In[18]:


data.describe()


# In[19]:


#to check for duplicates
idsUnique = len(set(data.Item_Identifier))
idsTotal = data.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")


# In[20]:


print(train.shape, test.shape, data.shape)


# In[21]:


#to check null values
data.apply(lambda x: sum(x.isnull()))


# In[22]:


#no: of unique values in each
data.apply(lambda x: len(x.unique()))


# In[23]:


#distribution of the target variable Item_Outlet_Sales
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(train.Item_Outlet_Sales, bins = 25)
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")


# In[24]:


print("Skew is:", train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())


# In[25]:


#to find correlation between numeric predictors and target variable
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[26]:


corr =numeric_features.corr()
corr


# In[27]:


print(corr['Item_Outlet_Sales'].sort_values(ascending=False))


# In[28]:


#correlation matrix
f, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, vmax=.8, square=True);


# In[29]:


#to find distribution with categorical predictors Item_Fat_Content
sns.countplot(train.Item_Fat_Content)


# In[30]:


#distribution of the variable Outlet_Size
sns.countplot(train.Outlet_Size)


# In[31]:


# distribution of the variable Outlet_Type
sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)


# In[32]:


#distribution of the variable Outlet_Location_Type
sns.countplot(train.Outlet_Location_Type)


# In[33]:


#distribution of the variable Item_Type
sns.countplot(train.Item_Type)
plt.xticks(rotation=90)


# In[34]:


#analysis between Item_weight and Item_Outlet_Sales
plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"],'.', alpha = 0.4)


# In[35]:


#analysis between item_visibility and Item_Outlet_sales
plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Visibility, train["Item_Outlet_Sales"],'.', alpha = 0.3)


# In[36]:


Item_Type = train.pivot_table(index=['Item_Type'], values="Item_Outlet_Sales", aggfunc=np.median)
Item_Type.plot(kind='bar', color='purple',figsize=(5,4))
plt.xlabel("Item_Type")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Item_Type on Item_Outlet_Sales")
plt.xticks(rotation=90)
plt.show()


# In[37]:


Item_Type = train.pivot_table(index=['Item_Type'], values="Item_Visibility", aggfunc=np.median)
Item_Type.plot(kind='bar', color='turquoise',figsize=(5,4))
plt.xlabel("Item_Type")
plt.ylabel("Sqrt Item_Visibility")
plt.title("Impact of Item_Type on Item_Visibility")
plt.xticks(rotation=90)
plt.show()


# In[38]:


Outlet_Establishment_Year_pivot = train.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='violet',figsize=(5,4))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[39]:


Item_Fat_Content_pivot = train.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='green',figsize=(5,4))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[40]:


train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())


# In[41]:


train.pivot_table(values='Outlet_Type', columns='Outlet_Size',aggfunc=lambda x:x.mode())


# In[42]:


train.pivot_table(values='Outlet_Location_Type', columns='Outlet_Type',aggfunc=lambda x:x.mode())


# In[43]:


Outlet_Identifier_pivot = train.pivot_table(index='Outlet_Identifier', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue',figsize=(5,4))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[44]:


Outlet_Size_pivot = train.pivot_table(index='Outlet_Size', values='Item_Outlet_Sales', aggfunc=np.median)
Outlet_Size_pivot.plot(kind='bar', color='purple',figsize=(5,4))
plt.xlabel('Outlet_Size')
plt.ylabel('Item_Outlet_Sales')
plt.title('Impact of Outlet_Size on Item_Outlet_Sales')
plt.xticks(rotation=0)
plt.show()


# In[45]:


Outlet_Type_pivot = train.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Type_pivot.plot(kind='bar', color='violet',figsize=(5,4))
plt.xlabel("Outlet_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[46]:


Outlet_Location_Type_pivot = train.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='magenta',figsize=(5,4))
plt.xlabel("Outlet_Location_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[47]:


#Data pre-processing


# In[48]:


#let do grouping in each catogorical columns

col=["Item_Fat_Content","Item_Type","Outlet_Location_Type","Outlet_Size"]

for i in col:
    print("The frequency distribution of each catogorical columns is--" + i+"\n")
    print(data[i].value_counts()) 


# In[49]:


#Replacing the minimum nan values in the Item_Weight with its mean value

#data.fillna({"Item_Weight":data["Item_Weight"].mean()},inplace=True)
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
#print(item_avg_weight)


# In[50]:


#checking the current status of  nan values in the dataframe
#nan_descript=data.apply(lambda x: sum(x.isnull()))
#Now we have 0 nan valuesin Item_Weight
data[:][data['Item_Identifier'] == 'DRI11']
def impute_weight(cols):
    Weight = cols[0]
    Identifier = cols[1]
    
    if pd.isnull(Weight):
        return item_avg_weight['Item_Weight'][item_avg_weight.index == Identifier]
    else:
        return Weight
print ('Orignal #missing: %d'%sum(data['Item_Weight'].isnull()))
data['Item_Weight'] = data[['Item_Weight','Item_Identifier']].apply(impute_weight,axis=1).astype(float)
print ('Final #missing: %d'%sum(data['Item_Weight'].isnull()))


# In[51]:


#data["Outlet_Size"].fillna(method="ffill",inplace=True)
#Import mode function:
from scipy.stats import mode
#Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
outlet_size_mode


# In[52]:


def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size
print ('Orignal #missing: %d'%sum(data['Outlet_Size'].isnull()))
data['Outlet_Size'] = data[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
print ('Final #missing: %d'%sum(data['Outlet_Size'].isnull()))


# In[53]:


#Creates pivot table with Outlet_Type and the mean of Item_Outlet_Sales. Agg function is by default mean()
data.pivot_table(values='Item_Outlet_Sales', columns='Outlet_Type')


# In[54]:


nan_descript=data.apply(lambda x: sum(x.isnull()))


# In[55]:


nan_descript


# In[56]:


data.head()


# In[57]:


#Now working on the item_visibility
visibilty_avg=data.pivot_table(values="Item_Visibility",index="Item_Identifier")


# In[58]:


#Remember the data is from 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[59]:


itm_visi=data.groupby('Item_Type')


# In[60]:


data_frames=[]
for item,item_df in itm_visi:
   data_frames.append(itm_visi.get_group(item))
for i in data_frames:
    i["Item_Visibility"].fillna(value=i["Item_Visibility"].mean(),inplace=True)
    i["Item_Outlet_Sales"].fillna(value=i["Item_Outlet_Sales"].mean(),inplace=True)



# In[61]:


new_data=pd.concat(data_frames)

nan_descript=new_data.apply(lambda x: sum(x.isnull()))


# In[62]:


#Now we have successfully cleaned our complete dataset.
new_data["Item_Fat_Content"].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace=True)


# In[63]:


new_data["Item_Fat_Content"].value_counts()


# In[64]:


#Get the first two characters of ID:
#data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
#data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
#data['Item_Type_Combined'].value_counts()
#Mark non-consumables as separate category in low_fat:
#new_data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
#new_data['Item_Fat_Content'].value_counts()


# In[65]:


#to find distribution with categorical predictors Item_Fat_Content
sns.countplot(new_data.Item_Fat_Content)


# In[66]:


#feature engineering


# In[67]:


#Implementing one-hot-Coding method for getting the categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type','Outlet_Type']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[68]:


#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type'])


# In[69]:


#Exporting the datas

train = data.loc[data['source']=="train"]

test = data.loc[data['source']=="test"]


# In[70]:


from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


# In[71]:


#Drop unnecessary columns
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)


# In[72]:


#dropping Item_Outlet_Sales for building the prediction model
train.drop(['source'],axis=1,inplace=True)


# In[73]:


#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


# In[74]:


#testing different models


# In[75]:


train=pd.read_csv('train_modified.csv')
test=pd.read_csv('test_modified.csv')


# In[76]:


#let us keep a baseline model for non-predicting model
#MEAN BASED
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

import numpy as np
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g"  % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
     #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


# In[77]:


X_train = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
Y_train = train['Item_Outlet_Sales']
X_test = test.drop(['Item_Identifier','Outlet_Identifier'],axis=1).copy()

#X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)


# In[78]:


#Linear Regression Model
print("Creating the models and processing")
from sklearn.linear_model import LinearRegression, Ridge
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients') 

    


# In[79]:


alg1_accuracy = round(alg1.score(X_train,Y_train) * 100,2)
alg1_accuracy


# In[80]:


#Ridge Regression Model
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficient') #model coefficient
print("Model has been successfully created and trained. The predicted result is in alg2.csv")

alg2_accuracy = round(alg2.score(X_train,Y_train) * 100,2)
alg2_accuracy


# In[81]:


# Decision Tree Model

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=6, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances') #feature importances

print("Model has been successfully created and trained. The predicted result is in alg3.csv")
alg3_accuracy = round(alg3.score(X_train,Y_train) * 100,2)
alg3_accuracy


# In[82]:


#change parametre
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg4 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=70)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')

print("Model has been successfully created and trained. The predicted result is in alg3.csv")
alg4_accuracy = round(alg4.score(X_train,Y_train) * 100,2)
alg4_accuracy


# In[83]:


#Random Forest Model

from sklearn.ensemble import RandomForestRegressor

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=350,max_depth=7, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')#FI

print("Model has been successfully created and trained. The predicted result is in alg5.csv")

alg5_accuracy = round(alg5.score(X_train,Y_train) * 100,2)
alg5_accuracy


# In[84]:


# change parametre
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')
plt.show()

alg6_accuracy = round(alg6.score(X_train,Y_train) * 100,2)
alg6_accuracy


# In[85]:


#AdaBoost Model
from sklearn.ensemble import AdaBoostRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg7= AdaBoostRegressor(n_estimators=2000, learning_rate=0.05)
modelfit(alg7, train, test, predictors, target, IDcol, 'alg7.csv')
coef7= pd.Series(alg7.feature_importances_, predictors).sort_values(ascending=False)
coef7.plot(kind='bar', title='Feature Importances')#ft

alg7_accuracy = round(alg7.score(X_train,Y_train) * 100,2)
alg7_accuracy


# In[86]:


#Gradient Boost Model
from sklearn.ensemble import GradientBoostingRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg8 = GradientBoostingRegressor(n_estimators= 50, learning_rate= 0.03, max_depth= 4)
modelfit(alg8, train, test, predictors, target, IDcol, 'alg8.csv')
coef8 = pd.Series(alg8.feature_importances_, predictors).sort_values(ascending=False)
coef8.plot(kind='bar', title='Feature Importances')#FT

alg8_accuracy = round(alg8.score(X_train,Y_train) * 100,2)
alg8_accuracy


# In[93]:


#XGBoost Model

from xgboost import XGBRegressor
import xgboost as xgb
alg9 = XGBRegressor(n_estimators=1000, learning_rate=0.04)
alg9.fit(train[predictors], train[target],early_stopping_rounds=5,eval_set=[(test[predictors], test[target])], verbose=False)

#Predict training set:
train_predictions = alg9.predict(train[predictors])

# make predictions
predictions = alg9.predict(test[predictors])

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test[target])))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train[target]).values, train_predictions)))

#Export submission file:
test[target] = alg9.predict(test[predictors])
IDcol.append(target)
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv('alg9.csv', index=False)

#graph
coef9 = pd.Series(alg9.feature_importances_, predictors).sort_values(ascending=False)
coef9.plot(kind='bar', title='Feature Importances')#Ft
plt.show()

print("Model has been successfully created and trained. The predicted result is in alg9.csv")

alg9_accuracy = round(alg9.score(X_train,Y_train) * 100,2)
alg9_accuracy


# In[94]:


#validation
cv_score = cross_validation.cross_val_score(alg9, X_train,Y_train, cv=20, scoring='mean_squared_error')


# In[95]:


cv_score = np.sqrt(np.abs(cv_score))


# In[90]:


alg9_submission = pd.DataFrame({
    'Item_Identifier':test['Item_Identifier'],
    'Outlet_Identifier':test['Outlet_Identifier'],
    'Item_Outlet_Sales': predictions
},columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])


# In[91]:


alg9_submission.to_csv('submission.csv',index=False)

