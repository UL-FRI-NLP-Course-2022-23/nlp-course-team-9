import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from tqdm.auto import tqdm


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super(MyDataSet).__init__(),

        # TODO: read 3rd_try.pkl instead
        self.df = pd.DataFrame(pd.read_pickle(data_path))
        # self.shuffle_df()
        self.df.reset_index(drop=True, inplace=True)
    
    def shuffle_df(self):
        self.df = self.df.sample(frac=1)
        self.df.reset_index(drop=True,inplace=True)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,indx):
        return {
            "input": self.df.iloc[indx,:][0].strip(),
            "output": self.df.iloc[indx,:][1].strip()
        }
    

if __name__ == "__main__":
    # for testing only
    mds = MyDataSet("data/3rd_try.pkl")
    print(mds)