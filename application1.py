from flask import Flask, request, render_template
import pickle
import pandas as pd

# Create Flask application
application1 = Flask(__name__)
app1 = application1

# Import Ridge Regressor model
ridge_model = pickle.load(open(r'D:\Udemy_ML_and_DL\Github_Project\ridge.pkl', 'rb'))
scalar=pickle.load(open(r'D:\Udemy_ML_and_DL\Github_Project\scalar.pkl','rb'))

@app1.route("/")
def index():
    return render_template('index.html')
@app1.route('/predictdata',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        scaled_data=scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(scaled_data)
        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app1.run(debug=True)
