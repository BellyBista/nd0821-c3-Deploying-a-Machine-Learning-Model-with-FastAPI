"""
Unit test of main.py API module with pytest
author: Quadri Bello
Date: May 26th 2023
"""

from fastapi.testclient import TestClient
#from fastapi import HTTPException
import json
import logging

from main import app

client = TestClient(app)


def test_root():
    """
    Test welcome message for get at root
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to our model API"


def test_inference():
    """
    Test model inference output
    """
    sample =  {  'age':50,
                'workclass':"Private", 
                'fnlgt':234721,
                'education':"Doctorate",
                'education_num':16,
                'marital_status':"Separated",
                'occupation':"Exec-managerial",
                'relationship':"Not-in-family",
                'race':"Black",
                'sex':"Female",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':50,
                'native_country':"United-States"
            }

    data = json.dumps(sample)

    r = client.post("/inference/", data=data )

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == 50
    assert r.json()["fnlgt"] == 234721

    # test prediction vs expected label
    logging.info(f'********* prediction = {r.json()["prediction"]} ********')
    assert r.json()["prediction"][0] == '<=50K'


def test_inference_class0():
    """
    Test model inference output for class 0
    """
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

    r = client.post("/inference/", data=data )

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == 43
    assert r.json()["fnlgt"] == 292175

    # test prediction vs expected label
    logging.info(f'********* prediction = {r.json()["prediction"]} ********')
    assert r.json()["prediction"][0:] == '>50K'


def test_wrong_inference_query():
    """
    Test incomplete sample does not generate prediction
    """
    sample =  {  'age':50,
                'workclass':"Private", 
                'fnlgt':234721,
            }

    data = json.dumps(sample)
    r = client.post("/inference/", data=data )

    assert 'prediction' not in r.json().keys()
    logging.warning(f"The sample has {len(sample)} features. Must be 14 features")
        
    
if '__name__' == '__main__':
    test_root()
    test_inference()
    test_inference_class0()
    test_wrong_inference_query()
