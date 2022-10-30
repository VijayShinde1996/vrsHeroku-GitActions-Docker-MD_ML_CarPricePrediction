from flask import Flask,render_template,request,jsonify
import numpy as np
import joblib

APP17 = Flask(__name__)
#Lets Load the Pickle files -
scalling = joblib.load('scalling_md17.pkl')
model = joblib.load('train_md17.pkl')

@APP17.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@APP17.route('/predict_REST',methods=['POST'])
def predict_REST():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalling.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@APP17.route('/predict',methods=['POST'])
def predict():
    data=[float(i) for i in request.form.values()]
    final_input=scalling.transform(np.array(data).reshape(1,-1))
    output=round(model.predict(final_input)[0],2)
    return render_template("index.html",prediction_text="The Predicted Car Price is: {} Lakhs".format(output))

if __name__ == ('__main__'):
    APP17.run(debug = True)