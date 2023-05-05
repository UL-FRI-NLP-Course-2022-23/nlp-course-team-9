import datetime
import os
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.distributed as dist
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
        # Parameters
        random_seed = 1
        model_type = "cjvt/t5-sl-small" # or "cjvt/t5-sl-large" if we set up parallelization
        device = "cuda"
        lr = 0.001
        epochs = 1
        batch_size = 4
        num_workers = 10
        prefetch_factor = 5

        train_dataset, validation_dataset, test_dataset = prepare_datasets("./data/3rd_try.pkl")

        if torch.cuda.get_device_name(device) == "Tesla V100S-PCIE-32GB":
            cluster_start = datetime.datetime.now().replace(microsecond=0)
            from slurm.tg_status import send_status
            send_status(f"Training started\n"
                        f"batch size: {batch_size}\n"
                        f"train dataset size: {len(train_dataset)}\n"
                        f"validation dataset size: {len(validation_dataset)}\n"
                        f"test dataset size: {len(test_dataset)}")

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                       num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=False)

        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=2, prefetch_factor=2, pin_memory=False)

        val_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=2, prefetch_factor=2, pin_memory=False)

        model = transformers.T5ForConditionalGeneration.from_pretrained(model_type).to(device)

        # parallelization
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        # dist.init_process_group("nccl", rank=2, world_size=2) # for 2 GPUs
        # ddp_model = nn.parallel.DistributedDataParallel(model)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        tokenizer = transformers.T5Tokenizer.from_pretrained(model_type)

        for epoch in range(epochs):
            with tqdm(total=len(train_dataset), unit="batch") as pbar:
                for batch in train_dataloader:
                    inputs = batch["input"]
                    outputs = batch["output"]

                    input_ids = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
                    output_ids = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)

                    # outputs = ddp_model(input_ids=input_ids, labels=output_ids)
                    outputs = model(input_ids=input_ids, labels=output_ids)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()

                    pbar.update(len(inputs))
                    pbar.set_postfix(epoch=epoch, loss=loss.item())


            # Validation loop
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch["input"]
                    outputs = batch["output"]

                    input_ids = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
                    output_ids = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)

                    outputs = model(input_ids=input_ids, labels=output_ids)
                    val_loss += outputs.loss.item() #! TODO: preveri, če je loss izračunan kot povprečje batcha ali je seštevek vseh samplov v batchu

            val_loss /= len(validation_dataset) #! TODO: to vrne število vseh samplov (zna biti narobe, če je)

            epoch_status = f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}"
            print(epoch_status)
            if cluster_start: send_status(epoch_status)


    except Exception as e:
        nok_str = f"Training failed\n{e}"
        if cluster_start: send_status(nok_str)
        print(nok_str)
    else:
        ok_str = "Training finished"
        finish = datetime.datetime.now().replace(microsecond=0)
        dir_name = finish.isoformat()[:-3]
        model.save_pretrained(f"/d/hpc/projects/FRI/team9/models/{dir_name}")
        if cluster_start:
            ok_str += f", took {finish - cluster_start}"
            send_status(ok_str)
        print(ok_str)

# conda activate ???
# requirements.txt
# v filu extra/nlp-course-team-9/slurm/salloc_gpu_6h.sh je ukaz za rezervacijo
# ...če želiš samo pognat, napišeš srun z istimi paraamteri in dopišeš ukaz ki ga želiš da se požene
# "squeue --me" to pokaže kaj imaš rezervirano
# ssh <ime_noda_ki_si_si_ga_rezerviral_alociral> --> premakne te v ta node (pomembno da pogledaš, da se spet prijaviš v environment - conda activate ???)
# 
