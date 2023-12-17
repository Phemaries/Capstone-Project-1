import requests


url = 'http://localhost:9696/predict'

# 'age', 'gender', 'cholesterol','glucose', 'smoke', 'alcohol_consumption', 'active', 'bmi_class', 'blood pressure'
patient_id = 1

patient = {
  "age": 32,
  "gender": 1,
  "cholesterol": 2,
  "glucose": 2,
  "smoke": 0,
  "alcohol_consumption": 1,
  "active": 0,
  "bm_class": 1, 
  "blood_pressure": "High BP Stage 2"
}


response = requests.post(url, json=patient).json()
print(response)

if response['cardio_concern'] == True:
    print(f'Patient {patient_id} has a high chance of having a cardiovascular disease')
else:
    print(f'Patient {patient_id} is not under threat')