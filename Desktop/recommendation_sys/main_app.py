import csv

import pandas as pd
import numpy as np
from flask import Flask, request, render_template, url_for, redirect, send_file, make_response
from flask import send_from_directory, current_app
from werkzeug.utils import secure_filename
import os
import pickle
import sklearn


UPLOAD_FOLDER = 'C:/Users/SAITEJA/Desktop/recommendation_sys/folder'
# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route("/")
def index(user=None):
    return render_template("homepage.html", user=user)


@app.route("/get_customer_details", methods=['GET', 'POST'])
def customer_details():
    return render_template("customer_details.html")


@app.route("/train_model", methods=['GET', 'POST'])
def train_model():
    return render_template("train_model.html")


@app.route("/predicted_results", methods=['GET', 'POST'])
def predicted_results():

    # getting customer ID
    if request.method == 'POST':
        customer_id = request.form['CustomerID']

    x_test_csv_file = csv.reader(open(os.path.join(app.config['UPLOAD_FOLDER'], 'x_test_new.csv')), delimiter=",")   #mortgage
    x_test_overdraft = csv.reader(open(os.path.join(app.config['UPLOAD_FOLDER'], 'x_test_overdraft.csv')), delimiter=",")   #overdraft
    x_test_credit = csv.reader(open(os.path.join(app.config['UPLOAD_FOLDER'], 'x_test_credit.csv')), delimiter=",")  # credit
    x_test_savings = csv.reader(open(os.path.join(app.config['UPLOAD_FOLDER'], 'x_test_savings.csv')), delimiter=",") #savings
    test_original_csv_file = csv.reader(open(os.path.join(app.config['UPLOAD_FOLDER'], 'test_set_original.csv')), delimiter=",")

    # fetching details of customer from original_test_set
    for row2 in test_original_csv_file:
        if customer_id == row2[1]:
            print(row2)
            break

    # fetching details of customer from x_test
    for row1 in x_test_csv_file:
        if customer_id == row1[1]:
            print(row1)
            break

    # fetching details of customer from x_test
    for row_overdraft in x_test_overdraft:
        if customer_id == row_overdraft[1]:
            print(row_overdraft)
            break

    # fetching details of customer from x_test
    for row_credit in x_test_credit:
        if customer_id == row_credit[1]:
            print(row_credit)
            break

    # fetching details of customer from x_test
    for row_savings in x_test_savings:
        if customer_id == row_savings[1]:
            print(row_savings)
            break

    # prediction of mortgage
    data = row1[2:]
    data = list(map(float, data))
    model = pickle.load(open('lr1_model_mortgage.sav', 'rb'))
    var = model.predict([data])
    if var == 0:
        var = "No"
    else:
        var = "Yes"
    predicted_output = model.predict_proba([data])[:, 1]
    print(predicted_output)
    predicted_output = int(predicted_output[0]*100)
    # predicted_mortgage = (predicted_output >= 80.00)
    # print(predicted_output)

    # prediction of overdraft
    data_overdraft = row_overdraft[2:]
    data_overdraft = list(map(float, data_overdraft))
    model1 = pickle.load(open('logistics_regression_model_overdraft.sav', 'rb'))
    var1 = model1.predict([data_overdraft])
    if var1 == 0:
        var1 = "No"
    else:
        var1 = "Yes"
    predicted_output1 = model1.predict_proba([data_overdraft])[:, 1]
    print(predicted_output1)
    predicted_output1 = int(predicted_output1[0] * 100)
    # predicted_mortgage = (predicted_output >= 80.00)
    # print(predicted_output)

    # prediction of credit
    data_credit = row_credit[2:]
    data_credit = list(map(float, data_credit))
    model2 = pickle.load(open('logistics_regression_model_credit.sav', 'rb'))
    var2 = model2.predict([data_credit])
    if var2 == 0:
        var2 = "No"
    else:
        var2 = "Yes"
    predicted_output2 = model2.predict_proba([data_credit])[:, 1]
    print(predicted_output2)
    predicted_output2 = int(predicted_output2[0] * 100)
    # predicted_mortgage = (predicted_output >= 80.00)
    # print(predicted_output)

    # prediction of savings
    data_savings = row_savings[2:]
    data_savings = list(map(float, data_savings))
    model3 = pickle.load(open('svm_model_savings1.sav', 'rb'))
    var3 = model3.predict([data_savings])
    predicted_output3 = model3.predict_proba([data_savings])[:, 1]
    print(predicted_output3)
    predicted_output3 = int(predicted_output3[0] * 100)
    print(predicted_output3)

    return render_template("predicted_results.html", array_details=row2, mortgage=var, overdraft=var1, credit=var2,
                           savings=var3[0],
                           score=predicted_output, score_overdraft=predicted_output1, score_credit=predicted_output2,
                           score_savings=predicted_output3)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        file = request.files['file']
        filename = secure_filename(file.filename)
        # global var
        # var = filename
        # df.to_csv(np.os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # global flag
        # flag = "file_upload"
        return render_template("train_model.html")


if __name__ == "__main__":
    app.run(debug=True)






