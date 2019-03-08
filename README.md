# Bayesian_Test
Our first Python code for Bayes

##Linear Regression##
**Enter data**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
%matplotlib inline
wheat_pro = pd.read_csv('wheat_producers15.csv')
wheat_pro.head()
wheat_pro.info()
wheat_pro.describe()
wheat_pro.columns
**plotting data**
sns.pairplot(wheat_pro)
sns.distplot(wheat_pro['Area_harvested'])
sns.distplot(wheat_pro['Production'])
**correlation**
wheat_pro.corr()
**linear regression**
X = wheat_pro[['Area_harvested', 'Production']]
y = wheat_pro['Yield ']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
regr = linear_model.LinearRegression()
print('Coefficients: \n', lm.coef_)
from sklearn.metrics import mean_squared_error, r2_score
y_pred = lm.predict(X_test)
print("Mean squared error: %.2f"
    % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
