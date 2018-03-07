import pandas as pd
import quandl as Quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle

df = Quandl.get('WIKI/GOOGL', authtoken = "uSFM84UsENP7k3X9z1Zi")
#with open('df.pickle', 'wb') as f:
#	pickle.dump(df, f)

#f = open('df.pickle', 'rb')
#df = pickle.load(f)

df=df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]

df['pc_hl'] = ((df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']) * 100
df['pc_change'] = ((df['Adj. Close']-df['Adj. Open']) / df['Adj. Open']) * 100

df = df[['Adj. Close', 'pc_hl', 'pc_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
#df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X_old = np.array(df.drop(['label'],1))
X_old = preprocessing.scale(X_old)
X = X_old[:-forecast_out]
X_lately = X_old[-forecast_out:]
df.dropna(inplace = True)
y = np.array(df['label'])

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.1)

X_train = X
y_train = y
X_test = X_lately

clf = LinearRegression()
clf.fit(X_train, y_train)

#with open('LinearRegression.pickle', 'wb') as f:
#	pickle.dump(clf, f)

#pickle_in = open('LinearRegression.pickle', 'rb')
#clf = pickle.load(pickle_in)

#accuracy = clf.score(X_test, y_test)
y_test = clf.predict(X_test)
print(forecast_out)
print(y_test)
