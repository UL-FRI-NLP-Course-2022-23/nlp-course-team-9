import datetime
import os
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from altair.vega import data
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import custom_dataset

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Parameters
random_seed = 1
model_type = "cjvt/t5-sl-small" # or "cjvt/t5-sl-large"


def prepare_datasets(data_path, train_size=0.6, validation_size=0.2, test_size=0.2):
    if train_size + validation_size + test_size != 1.0:
        raise Exception("Train, validation and test set percentages don't sum to 1.")

    df = pd.DataFrame(pd.read_pickle(data_path))

    train, validation_and_test = train_test_split(df, test_size=test_size, shuffle=True, random_state=random_seed)
    validation, test = train_test_split(validation_and_test, test_size=validation_size/(validation_size + test_size))

    train = custom_dataset.MyDataSet(train)
    validation = custom_dataset.MyDataSet(validation)
    test = custom_dataset.MyDataSet(test)

    return train, validation, test


if __name__ == "__main__":
    cluster_start = False
    try:
        device = "cuda:0"
        lr = 0.001
        epochs = 5
        batch_size = 8

        train_dataset, validation_dataset, test_dataset = prepare_datasets("data/3rd_try.pkl")

        if torch.cuda.get_device_name(device) == "Tesla V100S-PCIE-32GB":
            cluster_start = datetime.datetime.now().replace(microsecond=0)
            from slurm.tg_status import send_status
            send_status(f"Training started\n"
                        f"batch size: {batch_size}\n"
                        f"train dataset size: {len(train_dataset)}\n"
                        f"validation dataset size: {len(validation_dataset)}\n"
                        f"test dataset size: {len(test_dataset)}")

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                       num_workers=2, prefetch_factor=2, pin_memory=False)

        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=2, prefetch_factor=2, pin_memory=False)

        val_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=2, prefetch_factor=2, pin_memory=False)

        model = transformers.T5ForConditionalGeneration.from_pretrained(model_type).to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        tokenizer = transformers.T5Tokenizer.from_pretrained(model_type)

        for epoch in range(epochs):
            with tqdm(total=len(train_dataset), unit="batch") as pbar:
                for batch in train_dataloader:
                    inputs = batch["input"]
                    outputs = batch["output"]

                    input_ids = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
                    output_ids = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)

                    outputs = model(input_ids=input_ids, labels=output_ids)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()

                    pbar.update(len(inputs))
                    pbar.set_postfix(epoch=epoch, loss=loss.item())

            if cluster_start: send_status(f"ran {epoch+1}/{epochs}\nloss: {loss.item()}")

    except Exception as e:
        nok_str = f"Training failed\n{e}"
        if cluster_start: send_status(nok_str)
        print(nok_str)
    else:
        ok_str = "Training finished"
        if cluster_start:
            cluster_finish = datetime.datetime.now().replace(microsecond=0)
            ok_str += f", took {cluster_finish-cluster_start}"
            send_status(ok_str)
        print(ok_str)
