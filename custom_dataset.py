import torch


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, df):
        super(MyDataSet).__init__()
        self.df = df

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
