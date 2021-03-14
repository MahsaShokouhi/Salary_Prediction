# coding=utf-8

from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import numpy as np
import pandas as pd
import shap


shap.initjs()


def model_predict(data_asarray):
    pass


app = Flask(__name__)


def model_predict(data_asarray):
    data_asframe = pd.DataFrame(data_asarray,
                                columns=['jobType',
                                         'degree',
                                         'major',
                                         'industry',
                                         'yearsExperience',
                                         'milesFromMetropolis'])
    return model.predict(data_asframe)


def force_plot_html(*args):
    force_plot = shap.force_plot(*args, matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "100%", "height": "200px", "border": 0})

# home webpage
@app.route('/')
def index():
    shap.initjs()
    return render_template('index.html')


# results webpage: predict salary for user input
@app.route('/predict', methods=['POST'])
def predict():
    shap.initjs()

    features_dict = request.form
    df = pd.DataFrame(features_dict, index=[0])

    prediction = model.predict(df)[0]
    output = round(prediction, 2)

    shap_values = explainer.shap_values(df)
    f_plot = shap.force_plot(explainer.expected_value, shap_values, df)

    return render_template('predict.html', x=features_dict,
                           prediction_text=f'Estimated salary: $ {output}',                           exp=f'<head>{shap.getjs()}</head><body>{f_plot.html()}</body>')


if __name__ == '__main__':
    shap.initjs()
    # load model
    model = pickle.load(open('../model/model.pkl', 'rb'))

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
    explainer = joblib.load('../model/explainer.bz2')

    app.run(host='0.0.0.0', port=8000, debug=True)