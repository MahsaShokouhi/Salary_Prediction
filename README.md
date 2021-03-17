# Salary Prediction Application
An Application to Predeict Salary for New Job Postings

### Table of Contents 

- [Business Problem](#Business-Problem)
- [Objective](#Objective)
- [Data](#Data)
- [Data Preprocessing](#Data-Preprocessing)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Models and Evaluation](#Models-and-Evaluation)
- [Error Analysis](#Error-Analysis)
- [Feature Importance](#Feature-Importance)
- [Prediction on Test Data](#Prediction-on-Test-Data)
- [Flask Web App](#Flask-Web-App)



## Business Problem
The main challenge for HR professionals in recruiting new employees is to find the suitable candidate with necessary skills. In order to stay competitive, attract the best candidates, and reduce employee turnover and the cost associated with it, it’s important to have an efficient recruitment process in place. An important aspect of effective recruitment is to have a good estimate of the salary for any job specification. However, salary often varies across industries, geographic location, etc., even for the same job specification, making it challenging to estimate.


## Objective
The goal was to build and examine predictive models on a set of job postings with salaries (train set) to find the best model (with smallest error) for salary prediction. The whole pipeline included data preprocessing and regression models, and was designed to make prediction for the test set as well as for a single record. The selected modelling pipeline was then used to predict salary for the test set. Finally, an application was developed to predict salary for the job specifications and the candidate profile supplied by the user.

## Data
The train data included salary and the job specifications, each with a unique identifier (jobId), for 1 million job postings. The job specifications were the same for the train and test sets, as listed below:
#### Categorical Features and Levels:
*	Degree: DOCTORAL, MASTERS, BACHELORS, HIGH_SCHOOL, NONE
*	Major: MATH, PHYSICS, CHEMISTRY, ENGINEERING, BUSINESS, COMPUTER SCIENCE, LITERATURE, BIOLOGY, NONE
*	Job-type: CEO, CFO, CTO, VICE-PRESIDENT, MANAGER, SENIOR, JUNIOR, JANITOR
*	Industry: HEALTH, WEB, AUTO, FINANCE, OIL, SERVICE, EDUCATION

#### Numerical Features:
*	Years’ Experience
*	Miles from Metropolis

## Data Preprocessing
[link to the data processing script](https://github.com/MahsaShokouhi/Salary_Prediction/blob/master/scripts/etl.py)
*	There were no duplicates in the train data based on the “jobId”. 
*	There were no missing data.
*	Records with invalid values for salary (salary = 0) were removed from the train set.
*	The data was checked for possible errors such as records with a Degree = HIGH_SCHOOL or NONE, but with a Major other than “NONE”.
*	Only 5 rows were removed from the train set after data cleaning.
*	Finally categorical variables were transfromed using one-hot encoding.


## Exploratory Data Analysis
* The target variable (salary) was not skewed; hence no transformation was required.

![figure1](/images/fig1.png)

* On average, employees working in “Oil” and “Finance” Industries earned highest salaries (regardless of the job-type), followed by “Web”, “Healthcare”, “Auto”, “Service”, and “Education” industries.

![figure2](/images/fig2.png)

* Similarly, employees with “Engineering” and “Business” majors earned highest salaries.

![figure3](/images/fig3.png)

* Ignoring the possible interactions and intercorrelations between variables, salary was most strongly correlated with ‘years’ experience’, ‘miles from metropolis, and ‘job-type’.

![figure4](/images/fig4.png)

## Models and Evaluation

[link to the model selection script](https://github.com/MahsaShokouhi/Salary_Prediction/blob/master/scripts/model_selection.py)

For comparison, and to establish a baseline, a simple model (dummy regressor) was created that predicted the average salary for all job postings.

To build and evaluate models, linear and non-linear models were examined. For each model, mean squared error (MSE) was averaged across 5-fold cross-validation.

The results and MSE for each model is listed below ([see the log file for model selection](https://github.com/MahsaShokouhi/Salary_Prediction/blob/master/log/model_selection.log)):

Model |	MSE
----- | ---
Baseline Model |	1499
Linear regression |	384
Linear regression with interaction | 367
Random forest	| 370

The best models with lowest MSE were Random forest and Linear regression with interaction between features. Random forest was selected for training ([link to the model training script](https://github.com/MahsaShokouhi/Salary_Prediction/blob/master/scripts/train.py)).

## Error Analysis
A closer look at the distribution of the actual and predicted salaries on the train set shows that overall, prediction was reasonably in line with actual salaries. However, the salaries in the range of 100-150 (approximately) were overestimated by the model, whereas higher salaries were underestimated.

![figure5](/images/fig5.png)

![figure6](/images/fig6.png)

![figure7](/images/fig7.png)

## Feature Importance
Among all the job specifications,  job-type, years’ experience, company’s distance from a major city (miles from metropolis), and the industry are the main factor in determining the salary.

![figure8](/images/fig8.png)

## Prediction on Test Data
[link to the prediction script](https://github.com/MahsaShokouhi/Salary_Prediction/blob/master/scripts/predict.py)

## Flask Web App
The app was designed to:
- Provide an overview of the training data used to build the model.
- Receive job specifications for a single job posting from the user. 
- Predict salary for job specifications and the candidate profile supplied by the user.
- Provide explanation about the contribution of each factor to the estimated salary and how salary was predicted by the model using SHAP.

*Running the app opens the "index" template:*

<br>

![figure9](/images/fig9.png)

<br>

*After selecting the job specification, the user is directed to the "predict" template. Here's and example:*

<br>

![figure10](/images/fig10.png)
