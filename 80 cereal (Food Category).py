
# coding: utf-8

# ## 80 cereals numerical
# 
# If you like to eat cereal, do yourself a favor and avoid this dataset at all costs. 
# After seeing these data it will never be the same for me to eat Fruity Pebbles again.
# Content
# 
# Fields in the dataset:
# 
#     Name: Name of cereal
#     mfr: Manufacturer of cereal
#         A = American Home Food Products
#         G = General Mills
#         K = Kelloggs
#         N = Nabisco
#         P = Post
#         Q = Quaker Oats
#         R = Ralston Purina 
#     type:
#         cold
#         hot 
# 
#     calories: calories per serving
#     protein: grams of protein
#     fat: grams of fat
#     sodium: milligrams of sodium
#     fiber: grams of dietary fiber
#     carbo: grams of complex carbohydrates
#     sugars: grams of sugars
#     potass: milligrams of potassium
#     vitamins: vitamins and minerals - 0, 25, or 100, indicating the typical percentage of FDA recommended
#     shelf: display shelf (1, 2, or 3, counting from the floor)
#     weight: weight in ounces of one serving
#     cups: number of cups in one serving
#     rating: a rating of the cereals (Possibly from Consumer Reports?)
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv("cereal.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[19]:


#df['type'].nunique()
df['type'].unique()


# In[8]:


#Name of cereal that have calorie count more than average
df[df['calories'] > 106]['mfr'].value_counts()


# Almost every mfr has calorie count more than average

# In[9]:


df['mfr'].unique()


# In[10]:


sns.boxplot(data=df,y='rating',x='shelf',hue='type')


# In[11]:


df1 = df[['calories', 'protein', 'fat', 'sodium', 'fiber',
       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups','rating']]
sns.pairplot(df1,palette = 'coolwarm')


# In[12]:


sns.jointplot(x = df['fat'], y = df['rating'])


# We can see here that more is the fat content lesser is the rating.

# In[13]:


df[df['rating'] > 42]['fat'].value_counts()


# In[14]:


df[df['rating'] > 42]['type'].value_counts()


# In[15]:


sns.jointplot(x = df['sugars'], y = df['rating'])


# In[16]:


sns.jointplot(x = df['calories'], y = df['rating'])


# In[17]:


plt.scatter(x = df['calories'], y = df['rating'])


# In[18]:


df[df['rating'] > 50]['type'].value_counts()


# In[20]:


mf = pd.get_dummies(df["mfr"], drop_first=True, prefix="mfr")
tp = pd.get_dummies(df["type"], drop_first=True, prefix="type")
shl = pd.get_dummies(df["shelf"], drop_first=True, prefix="shelf")


# In[21]:


df.drop(["name", "mfr", "type", "shelf"],axis=1,inplace=True)
df.head()


# In[22]:


df = pd.concat([df, mf, tp, shl],axis=1)
df.head()


# # Building a Linear Regression model

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('rating',axis=1), df['rating'], test_size=0.30,random_state=101)


# In[26]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[27]:


predictions = lm.predict(X_test)


# In[29]:


plt.scatter(y_test,predictions)
plt.xlabel('y_test')
plt.ylabel('prediction')
plt.title('Real vs Precited')


# ### Evaluating the model

# In[31]:


metrics.mean_absolute_error(y_test,predictions)


# In[32]:


metrics.mean_squared_error(y_test,predictions)


# In[33]:


np.sqrt(metrics.mean_absolute_error(y_test,predictions))


# In[38]:


metrics.r2_score(y_test, predictions)


# In[39]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[55]:


#Coefficient - Std. Error - tStatistic - pValue, Simple Linear Regression
est = smf.ols('rating ~ calories + protein + fat + sodium + fiber + carbo + sugars + potass + vitamins + weight + cups', df).fit()
est.summary().tables[1]


# In[44]:


est.summary()


# ### Modifications to increase the performance
# 
# removing mfr as it is having p-value greater than 0.05

# In[45]:


X = df[["calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass", "vitamins", "weight", "cups", "type_H", "shelf_2", "shelf_3"]]
y = df["rating"]


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('y_test')
plt.ylabel('prediction')
plt.title('Real vs Precited')


# In[50]:


print("MAE : ",metrics.mean_absolute_error(y_test,predictions))

print("MSE", metrics.mean_squared_error(y_test,predictions))

print("sqrt(MAE) : ", np.sqrt(metrics.mean_absolute_error(y_test,predictions)))

print("R2 score : ", metrics.r2_score(y_test, predictions))


# checking if after removing the shelf column, as it is having p < 0.05, its impact

# In[51]:


X = df[["calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass", "vitamins", "weight", "cups", "type_H", "shelf_2", "shelf_3"]]
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('y_test')
plt.ylabel('prediction')
plt.title('Real vs Precited')


# In[52]:


print("MAE : ",metrics.mean_absolute_error(y_test,predictions))

print("MSE", metrics.mean_squared_error(y_test,predictions))

print("sqrt(MAE) : ", np.sqrt(metrics.mean_absolute_error(y_test,predictions)))

print("R2 score : ", metrics.r2_score(y_test, predictions))


# the shelf column is hence not removed as errors increased and r2 decreased

# After removing the mfr columns, checking if after removing the weight column, as it is having p < 0.05, its impact

# In[53]:


X = df[["calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass", "vitamins", "cups", "type_H", "shelf_2", "shelf_3"]]
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('y_test')
plt.ylabel('prediction')
plt.title('Real vs Precited')

print("MAE : ",metrics.mean_absolute_error(y_test,predictions))

print("MSE", metrics.mean_squared_error(y_test,predictions))

print("sqrt(MAE) : ", np.sqrt(metrics.mean_absolute_error(y_test,predictions)))

print("R2 score : ", metrics.r2_score(y_test, predictions))


# Weight column doen't play specific role in determining the ratings, hence it is now removed.
# 
# Now  let's continue with removing the cups columns also.

# In[56]:


X = df[["calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass", "vitamins", "type_H", "shelf_2", "shelf_3"]]
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('y_test')
plt.ylabel('prediction')
plt.title('Real vs Precited')

print("MAE : ",metrics.mean_absolute_error(y_test,predictions))

print("MSE", metrics.mean_squared_error(y_test,predictions))

print("sqrt(MAE) : ", np.sqrt(metrics.mean_absolute_error(y_test,predictions)))

print("R2 score : ", metrics.r2_score(y_test, predictions))


# the change is very minor so it's better to remove them.

# # Conclusion

# Analysis:
# 
# *	Cold food cereals are more preferred as compared to hot cereals(as per rating)
# *	Lesser the fat content in the cereals, more is the rating.
# *	Final predictors : type, calories, protein, fat, sodium, fiber, carbo, sugars, potass, vitamins, shelf...
# *   Target column : "Rating"
# 
# 

# MAE :  3.1475145615781724e-07
# MSE 1.370994482800535e-13
# sqrt(MAE) :  0.0005610271438690086
# R2 score :  0.9999999999999993

# ### Thannk You.
# 
# ### Keep practicing more and upgrade yourself
