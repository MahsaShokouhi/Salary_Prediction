# coding=utf-8

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# load model
model = joblib.load('../model/model.lzma')


def model_predict(data_asarray):
    data_asframe = pd.DataFrame(data_asarray,
                                columns=['jobType',
                                         'degree',
                                         'major',
                                         'industry',
                                         'yearsExperience',
                                         'milesFromMetropolis'])
    return model.predict(data_asframe)


# load SHAP explainer
explainer = joblib.load('../model/explainer.lzma')


# home webpage
@app.route('/')
def index():
    return render_template('index.html')


# results webpage: predict salary for user input
@app.route('/predict', methods=['POST'])
def predict():
    features_dict = request.form
    df = pd.DataFrame(features_dict, index=[0])

    prediction = model.predict(df)
    output = round(prediction[0], 2)

    shap_values = explainer.shap_values(df)
    f_plot = shap.force_plot(explainer.expected_value, shap_values, df)

    return render_template('predict.html', x=features_dict,
                           prediction_text=f'Estimated salary: $ {output}',
                           exp=f'<head>{shap.getjs()}</head><body>{f_plot.html()}</body>')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
