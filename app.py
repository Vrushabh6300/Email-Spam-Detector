# from logging import debug
from flask import Flask
from flask import render_template, request, flash, redirect, url_for
import numpy as np
import pickle
import model

app = Flask(__name__)
app.secret_key = 'NaiveBayes'
prd = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

# def showResult(result):
#     if(result == 'spam'):
#         flash('This E-mail is a spam!...', 'danger')
#     else:
#         flash('This E-mail is not a spam...', 'success')
#     return render_template('home.html', result=result)


@app.route('/getTestMail', methods = ['POST', 'GET'])
def getTestMail():
    if request.method == 'POST':
        testMail = request.form['test-mail']
        msgs = []
        msgs.append(testMail)
        messages = model.prepare(msgs)
        y_pred = prd.predict(messages)
        print("Prediction: ", y_pred)
        result = y_pred[0]
        if(result == 'spam'):
            return render_template('result.html', title='Result', result=result, testMail=testMail)
        else:
            return render_template('result.html', title='Result', result=result, testMail=testMail)
        
    else:
        testMail = request.args.get('test-mail')
        return "Test Mail: " + testMail


if __name__ == '__main__':
    app.run(debug=True)