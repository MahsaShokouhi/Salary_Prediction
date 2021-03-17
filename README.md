# Salary_Prediction
Salary Prediction for the Job Postings

### Table of Contents  
[Heading](#Business Problem)  


## Business Problem
The main challenge for HR professionals in recruiting new employees is to find the suitable candidate with necessary skills. In order to stay competitive, attract the best candidates, and reduce employee turnover and the cost associated with it, it’s important to have an efficient recruitment process in place. An important aspect of effective recruitment is to have a good estimate of the salary for any job specification. However, salary often varies across industries, geographic location, etc., even for the same job specification, making it challenging to estimate.

## Objective
The goal was to use predictive models on a set of job postings with salaries (train set), find important factors in estimating salary, and examine which model best predicts the salary for any job specification. The best model was then used to predict salaries for a new set of job postings (test set).

## Data
The train data included salary and the job specifications, each with a unique identifier (jobId), for 1 million job postings. The job specifications were the same for the train and test sets, as listed below:
#### Categorical Features and Levels:
* Degree: DOCTORAL, MASTERS, BACHELORS, HIGH_SCHOOL, NONE
* Job-type: CEO, CFO, CTO, VICE-PRESIDENT, MANAGER, SENIOR, JUNIOR, JANITOR
*	Industry: HEALTH, WEB, AUTO, FINANCE, OIL, SERVICE, EDUCATION
*	Major: MATH, PHYSICS, CHEMISTRY, ENGINEERING, BUSINESS, COMPUTER SCIENCE, LITERATURE, BIOLOGY, NONE
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

[see the log file for data processing](https://github.com/MahsaShokouhi/Salary_Prediction/blob/master/log/etl.log)

## Exploratory Data Analysis
*	The target variable (salary) was not skewed; hence no transformation was required.

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

To build and evaluate models, a range of linear and tree-based models were examined. For each model, mean squared error (MSE) was averaged across 5-fold cross-validation.

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
Among all the job specifications, job-type was the most important factor in determining the salary, followed by “years of experience”, “miles from metropolis”, “industry”, and “major”.

![figure8](/images/fig8.png)

## Prediction on test set
[link to the prediction script](https://github.com/MahsaShokouhi/Salary_Prediction/blob/master/scripts/predict.py)

## Conclusion
To summarize, job-type, years’ experience, company’s distance from a major city (miles from metropolis), and the company’s industry are the main factor in determining the salary. However, each of these factors should not be considered in isolation, as the interaction between them is also important in determining the appropriate salary.
