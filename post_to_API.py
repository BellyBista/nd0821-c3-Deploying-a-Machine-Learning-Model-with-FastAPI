"""
Script to post to FastAPI instance for model inference
author: Quadri Bello
Date: May 26th 2023
"""

import requests
import json

#url = "enter heroku web app url here"
url = "https://python-app1.herokuapp.com/inference"


# explicit the sample to perform inference on
sample =  { 'age':43,
            'workclass':"Self-emp-not-inc", 
            'fnlgt': 292175,
            'education':"Masters",
            'education_num':14,
            'marital_status':"Divorced",
            'occupation':"Exec-managerial",
            'relationship':"Unmarried",
            'race':"White",
            'sex':"Female",
            'capital_gain':50000,
            'capital_loss':0,
            'hours_per_week':45,
            'native_country':"United-States"
            }

data = json.dumps(sample)

# post to API and collect response
response = requests.post(url, data=data )

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())

 
