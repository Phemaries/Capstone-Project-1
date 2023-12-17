# Cardiovascular Disease Prediction 

This repository contains EDA and prediction of present/absent cases of Cardiovascular disease in patients using various classification models to get the optimum accuracy for prediction. Also, Docker Containerization was achieved in this project.


## Why this Project?

According to the National Institute of Health, United States, there has been a steady rise in deaths and disability arising from cardiovascular disease over the last 30 years. In 2019 alone, the condition, which includes heart disease and stroke, was responsible for a staggering one-third of all deaths worldwide. This prompted my motive to contribute in this project.

## Data
The data (70,000 records) is sourced from Kaggle. Check the dataset [here](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data).

### Data Features

| Feature       | Description           | 
| ------------- |:-------------:| 
| Age     | In unit days| 
| Height     | Height in centimeters     |   
| Weight | Units in Kilograms      |  
| Gender | Male or Female     | 
| Systolic Blood Pressure (ap_hi) | Systolic pressure    | 
| Diastolic Blood Pressure (ap_lo) | Diastolic pressure      | 
| Cholesterol | cholesterol level    | 
| Glucose (gluc) | glucose level     | 
| Smoking (smoke) | Smoke or Don't Smoke   | 
| Alcohol Intake (alco) |Drink or Don't Drink   | 
| Physical activity (active) | Active or not  | 
| Presence or absence of cardiovascular disease (cardio) | Presence or absence  | 

## Additional Features
Additional features in relevance to the scope of this study were derived.

| Feature       | Description           | 
| ------------- |:-------------:| 
| Body Mass Index (BMI) | Classified as Overweight or Not  | 
| Blood Pressure | Categories of blood pressure derived from Systolic/Diastolic readings  | 

### Exploratory Data Analysis (EDA)
An extensive EDA was carried out on the dataset. Data had no null values and explored each feature against the target variable 'cardio_status' using data visualization and other metrics. This also provided some basic answers to providing a best fit for prediction. Non-ideal systolic and diastolic blood pressure values were removed (Total number of records now 68,781


### Model Training.
Multiple models were trained and compared for the dataset (60% training/ 20% validation / 20% testing). Models such as Logistic Regression, RandomForestClassifier, DecisionTreeClassifier, and RidgeClassifier with optimized parameters were selected to get the best metrics based on ##accuracy, ##confusion matrix, ##precision, ##recall, and ##f1_score. 

### Finetuning the models
All selected models were fine-tuned and the `Precison_Recall` curve was plotted for all the models. 

### Conclusion
Upon hypermetric tuning, the model `LogisticRegression(solver='liblinear', C= 1, max_iter= 100)` has an accuracy of 72.32%. Run on `train.py` file

### Model and Deployment to Flask
* The best model is saved into `modellr.pkl` with the `dv` and `modellr` features.
* waitress-serve --listen=0.0.0.0:9696 predict:app
* Create a virtual environment using: python -m venv env
* In the project directory run `env\Scripts\activate`
* Install all packages `pip install [packages]`
* `pip freeze > requirements.txt` to create `requirement.txt` file
* run `python predict_test.py`

### Containerization
* Build the Docker file: docker build -t cardio-prediction .
* To run it: docker run -it -p 9696:9696 cardio-prediction:latest


### Future Scope
*  Deployment to Cloud

