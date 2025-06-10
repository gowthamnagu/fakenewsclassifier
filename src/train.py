import pickle
import mlflow.sklearn
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow
import ast
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,confusion_matrix,classification_report

from mlflow.models import infer_signature
from urllib.parse import urlparse
import os

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/gowthambreeze/fakenewsclassifier.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']='gowthambreeze'
os.environ['MLFLOW_TRACKING_PASSWORD']='ca435b40d4aa473afdd3245c5c719bce854bc70d'

def hyperparameter_tuning(X_train,y_train,X_test,y_test,epochs,batch_size):
    embedding_vector_features=40 ##features representation
    voc_size=5000
    sent_length=20
    model=Sequential()
    model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
    model.add(LSTM(100))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size)
    return model

params=yaml.safe_load(open("params.yaml"))["train"]
def train_model(data_path,model_path,epochs,batch_size):
    data=pd.read_csv(data_path)
    X_final = data['title'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()
    X_final = np.array(X_final)
    y_final = data['label']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
        signature=infer_signature(X_train,y_train)
        

        final_model=hyperparameter_tuning(X_train,y_train,X_test,y_test,epochs,batch_size)

        y_pred=final_model.predict(X_test)
        y_pred=np.where(y_pred > 0.6, 1,0) ##AUC ROC Curve
        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)
        accuracy=accuracy_score(y_test,y_pred)
        ps=precision_score(y_test, y_pred)
        rs=recall_score(y_test, y_pred)
        f1=f1_score(y_test, y_pred)
        

        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("precision", ps)
        mlflow.log_metric("recall", rs)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(str(cr),"classificaiton_report.txt")

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(final_model,"model",registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(final_model, "model",signature=signature)

        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        filename=model_path
        pickle.dump(final_model,open(filename,'wb'))

        print(f"Model saved to {model_path}")

if __name__=="__main__":
    train_model(params['data'],params['model'],params['epochs'],params['batch_size'])
