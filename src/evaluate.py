import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
import yaml
import os
from urllib.parse import urlparse
import ast
import numpy as np


os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/gowthambreeze/fakenewsclassifier.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']='gowthambreeze'
os.environ['MLFLOW_TRACKING_PASSWORD']='ca435b40d4aa473afdd3245c5c719bce854bc70d'

#load parameter from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data=pd.read_csv(data_path)
    X = data['title'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()
    X = np.array(X)
    y=data['label']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    ##load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    predictions=np.where(predictions > 0.6, 1,0)
    accuracy=accuracy_score(y,predictions)

    ##log metrics to MLFLOW
    mlflow.log_metric("accuracy",accuracy)

if __name__=="__main__":
    evaluate(params['data'],params['model'])
