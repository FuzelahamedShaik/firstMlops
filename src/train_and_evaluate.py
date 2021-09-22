import pandas as pd
from sklearn.linear_model import ElasticNet
from get_data import read_config
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import argparse
import json

def evaluate_metrics(actual,predict):
    rmse = mean_squared_error(actual,predict)
    mae = mean_absolute_error(actual,predict)
    r_2 = r2_score(actual,predict)
    return (rmse,mae,r_2)

def train_and_evaluate(config_path):
    config = read_config(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    target = [config["base"]["traget_col"]]
    train = pd.read_csv(train_data_path,sep=",")
    test = pd.read_csv(test_data_path,sep=",")
    y_train = train[target]
    y_test = test[target]
    X_train = train.drop(target,axis=1)
    X_test = test.drop(target,axis=1)
    lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=random_state)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    (rmse,mae,r_2) = evaluate_metrics(actual=y_test,predict=y_pred)

    scores_path = config["reports"]["scores"]
    params_path = config["reports"]["params"]

    with open(scores_path,"w") as f:
        scores = {
            "RMSE" : rmse,
            "MAE" : mae,
            "R_2" : r_2
        }
        json.dump(scores,f,indent=4)

    with open(params_path,"w") as f:
        params = {
            "alpha" : alpha,
            "l1_ratio" : l1_ratio,
            "random_state" : random_state
        }
        json.dump(params,f,indent=4)

    print("rmse:%s, mae=%s, r_2=%s" %(rmse,mae,r_2))
    print("Dumping the model.............")

    #os.makedirs(model_dir,exist_ok=True)
    model_path = os.path.join(model_dir,"model.joblib")
    joblib.dump(lr,model_path)
    print("model is dumped into %s" %(model_path))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    config = args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)