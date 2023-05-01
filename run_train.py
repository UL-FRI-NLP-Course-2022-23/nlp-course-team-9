import datetime
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
import custom_dataset

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

if __name__ == "__main__":
    cluster_start = False
    try:
        device = "cuda:0"
        lr = 0.001
        epochs = 10
        batch_size = 32

        train_dataset = custom_dataset.MyDataSet("data/3rd_try.pkl")

        if torch.cuda.get_device_name(device) == "Tesla V100S-PCIE-32GB":
            cluster_start = datetime.datetime.now().replace(microsecond=0)
            from slurm.tg_status import send_status
            send_status(f"training started with\nbatch size: {batch_size}\ntrain dataset size: {len(train_dataset)}")

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                       num_workers=2, prefetch_factor=2, pin_memory=False)
        # train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

        # test_dataset = ???
        # test_dataloader = ???

        model = transformers.T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        # model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        # model = transformers.T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
        # model = transformers.T5ForConditionalGeneration.from_pretrained("t5-3b").to(device)
        # model = transformers.T5ForConditionalGeneration.from_pretrained("t5-11b").to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        tokenizer = transformers.T5Tokenizer.from_pretrained("t5-small")
        # tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base")
        # tokenizer = transformers.T5Tokenizer.from_pretrained("t5-large")
        # tokenizer = transformers.T5Tokenizer.from_pretrained("t5-3b")
        # tokenizer = transformers.T5Tokenizer.from_pretrained("t5-11b")

        for epoch in range(epochs):
            if cluster_start: send_status(f"running {epoch+1}/{epochs}")

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

            if cluster_start: send_status(f"loss: {loss.item()}")

    except Exception as e:
        nok_str = f"training failed\n{e}"
        if cluster_start: send_status(nok_str)
        print(nok_str)
    else:
        ok_str = "training finished"
        if cluster_start:
            cluster_finish = datetime.datetime.now().replace(microsecond=0)
            ok_str += f", took {cluster_finish-cluster_start}"
            send_status(ok_str)
        print(ok_str)
