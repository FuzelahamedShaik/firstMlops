import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from get_data import read_config, get_data

def split_and_save_data(config_path):
    config = read_config(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    split_ratio = config["split_data"]["test_size"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    random_state = config["base"]["random_state"]
    df = pd.read_csv(raw_data_path,sep=",")
    train, test = train_test_split(df,test_size=split_ratio,random_state=random_state)
    train.to_csv(train_data_path,sep=",",index=False)
    test.to_csv(test_data_path,sep=",",index=False)
    print("train and test data got pushed to their respective directories!!")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    config = args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)