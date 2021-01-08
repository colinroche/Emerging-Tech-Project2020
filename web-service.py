# Adapted from: https://flask.palletsprojects.com/en/1.1.x/quickstart/#a-minimal-application
from flask import Flask,redirect
import pandas as pd
import numpy as np
import sklearn.linear_model as sklm

app = Flask(__name__)

df = pd.read_csv("https://raw.githubusercontent.com/ianmcloughlin/2020A-machstat-project/master/dataset/powerproduction.csv")

X = df['speed']
y = df['power']

df = df.drop([208, 340, 404, 456, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499])

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values
X = X.reshape(-1,1)

model = sklm.LinearRegression()
model.fit(X,y)

p = model.predict(X)


@app.route('/')
def hello_world():
    return redirect('/table')

@app.route('/table')
def predictions(X):
    return "Input: %s " %X