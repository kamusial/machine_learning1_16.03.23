"""
 Udostępnianie modelu ML jako usługi API
"""
from flask import Flask, request   #w request są parametry
import joblib
import numpy as np

app = Flask("KNN")
# model = joblib.load("knn.model")
#
# @app.route("/predict")
# def predict():
#     try:
#         sl = float(request.args.get("sl",0))
#         sw = float(request.args.get("sw",0))
#         pl = float(request.args.get("pl",0))
#         pw = float(request.args.get("pw",0))
#         print(sl,sw,pl,pw)
#         if sl<=0 or sw<=0 or pl<=0 or pw<=0:
#             raise Exception("sprawdz parametry wejsciowe")
#
#         sample = np.array([ [sl,sw,pl,pw] ])
#         result = model.predict(sample)
#
#         iris = ["setosa", "versicolor", "virginica"]
#         return iris[ result[0] ]
#
#     except Exception as exc:
#         return str(exc)
#
# @app.route("/")
# def hello():
#     return "<h1>Hello KNN model</h1>"

app.run(port=1234, debug=True)