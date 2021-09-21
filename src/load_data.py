import os
from get_data import read_config, get_data
import yaml
import argparse
import pandas as pd

def load_and_save(config_path):
    config = read_config(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ","_") for col in df.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path,sep=",",index=False,header=new_cols)
    print(f"raw data is stored in the {raw_data_path} directory!!")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    config = args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)