import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import tensorflow as tf
from tensorflow import keras

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df=pd.read_csv('WineQT.csv')
print(df.head().T)
# print(ds.describe())
# print(ds.info())
print(df.isnull().sum())

df_x=df.drop(columns=['quality'])
df_y=df['quality']


reg=linear_model.LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.4,random_state=42)

mnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()
model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
model.compile(optimizer=tf.optimizers.Adam(),loss='spars_categorical_crossentropy',metrics=['acuracy'])

reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)


print(mean_squared_error(y_test,y_pred))


print("Height:",reg.coef_)
print("Intercept:",reg.intercept_)

plt.scatter(y_test,y_pred)
plt.stem(y_test,y_pred)
plt.plot(y_test,y_pred)
plt.show()
