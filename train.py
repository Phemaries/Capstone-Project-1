

#Import Libraries
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 500)

import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# function for blood pressure classification
def pressure_label(row):
    if row['ap_hi'] < 90 and row['ap_lo'] < 60:
        return 'Low Blood Pressure'
    elif row['ap_hi'] < 120 and row['ap_lo'] < 80:
        return "Normal Blood Pressure"
    elif row['ap_hi'] < 130 and row['ap_lo'] < 80:
        return "Elevated Blood Pressure"
    elif row['ap_hi'] < 140 and row['ap_lo'] < 90:
        return "High BP Stage 1"
    else:
        return "High BP Stage 2"
   


df = pd.read_csv("C:/Users/lenovo/Desktop/ML Project DataKlub/cardio_train.csv", sep=';')


df.rename(columns={"gluc":"glucose", "alco":"alcohol_consumption", "cardio":"cardio_status"}, inplace=True)





# Discarding blood pressure greater than 370/360 mmHg and blood pressure less than 50/20mmHg respectively
df = df.loc[(df["ap_hi"] > 50) & (df["ap_hi"] < 370) & (df["ap_lo"] > 20) & (df["ap_lo"] < 360)]
df.reset_index(inplace=True)


df['blood_pressure'] = df.apply(pressure_label, axis=1)


## change days to years in age column
df["age"] = df["age"].apply(lambda x: round(x/365))


df["bmi"] = df["weight"] *10000 / ((df["height"])**2)
df['bmi_class'] = df['bmi'].apply(lambda x : x < 25)
df['bmi_class']



## let's drop the id column which won't contribute to the model using domain knowldedge.
df.drop(['id', 'index'], axis=1, inplace=True)



## convert 
df[['weight', 'bmi_class']] = df[['weight', 'bmi_class']] .astype('int64')


cat_cols = list(df.dtypes[df.dtypes == 'object'].index)
print(f'Columns with categorical variables are: {cat_cols} \n')


int_cols = list(df.dtypes[df.dtypes == 'int64'].index)
print(f'Columns with integer values are: {int_cols}')



## for better computation, replace 1 with 0, 2 with 1, 3 with 2
df[['gender', 'glucose', 'cholesterol']] = df[['gender', 'glucose', 'cholesterol']].apply(lambda x : x-1)



select = ['gender', 'glucose', 'cholesterol', 'smoke', 'alcohol_consumption', 'active', 'bmi_class', 'blood_pressure', 'cardio_status']
df_select = df[select]

for col in df_select.columns:
    gra = df_select[col].value_counts()
    print(f'{col.upper()}: \n')
    print(gra)
    print('\n')
    


df = df[['age', 'gender', 'cholesterol','glucose', 'smoke', 'alcohol_consumption', 'active', 'blood_pressure', 'bmi_class', 'cardio_status']]


# split data set to 60/20/20 for validation and testing
data_full_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
data_train, data_val = train_test_split(data_full_train, test_size=0.25, random_state=42)


len(data_train), len(data_val), len(data_test)


y_train = data_train.cardio_status.values
y_val = data_val.cardio_status.values
y_test = data_test.cardio_status.values

del data_train['cardio_status']
del data_val['cardio_status']
del data_test['cardio_status']

numerical_cols = ['age', 'gender', 'cholesterol','glucose', 'smoke', 'alcohol_consumption', 'active', 'bmi_class']

categorical_cols = ['blood_pressure']

dicts_full_train = data_full_train[numerical_cols + categorical_cols].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

y_full_train = data_full_train.cardio_status.values


modellr = LogisticRegression(solver='liblinear', C= 1, max_iter= 100)
modellr.fit(X_full_train, y_full_train)


dicts_test = data_test[numerical_cols + categorical_cols].to_dict(orient='records')
X_test = dv.transform(dicts_test)

lr_pred = modellr.predict(X_test)
lr_class = classification_report(y_test, lr_pred)
lr_score = accuracy_score(y_test, lr_pred)
print(lr_class)
print(f'\n Accuracy score of {lr_score} with Logistic Regression')



with open('modellr.pkl', 'wb') as f_out:
    pickle.dump((dv, modellr), f_out)

print(f'the model is saved to modellr.pkl')
