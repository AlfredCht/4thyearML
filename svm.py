import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split


emp = pd.read_csv('Data.csv')
empquitte = emp.loc[emp['left']==1]

emp.loc[emp.salary == 'low','salary'] = 0
emp.loc[emp.salary == 'medium','salary'] = 1
emp.loc[emp.salary == 'high','salary'] = 2

emp.loc[emp.sales == 'IT','sales'] = 0
emp.loc[emp.sales == 'RandD','sales'] = 1
emp.loc[emp.sales == 'accounting','sales'] = 2
emp.loc[emp.sales == 'hr','sales'] = 3
emp.loc[emp.sales == 'management','sales'] = 4
emp.loc[emp.sales == 'marketing','sales'] = 5
emp.loc[emp.sales == 'product_mng','sales'] = 6
emp.loc[emp.sales == 'sales','sales'] = 7
emp.loc[emp.sales == 'support','sales'] = 8
emp.loc[emp.sales == 'technical','sales'] = 9

data = emp.select_dtypes(include=[np.number]).interpolate().dropna()

y = emp.left

X = emp.drop(["left","sales","Work_accident","promotion_last_5years","salary"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=.3, train_size = 0.7)

from sklearn import svm
clf1 = svm.SVR()
clf1.fit(X_train,y_train)
print('Résultat SVM : ')
print(clf1.score(X_test, y_test))
