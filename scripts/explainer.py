import pandas as pd
import shap
import joblib


train = pd.read_csv('../data/derived/train.csv')
x_train = train.drop(['salary'], axis=1)
# y_train = train['salary']

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


explainer = shap.KernelExplainer(model_predict, x_train)
joblib.dump(explainer, filename='explainer.lzma', compress=('lzma', 6))
