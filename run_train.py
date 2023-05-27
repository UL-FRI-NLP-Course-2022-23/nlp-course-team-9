import datetime
import os
import pandas as pd
import torch
import torch.nn as nn
import traceback
from transformers import T5ForConditionalGeneration, T5Tokenizer
from slurm.tg_status import send_status
from tqdm.auto import tqdm


if __name__ == "__main__":
    try:
        # General parameters
        random_seed = 1
        device = "cuda"
        num_cpus = len(os.sched_getaffinity(0))
        model_type = "cjvt/t5-sl-large"
        lr = 0.001
        epochs = 1

        # DataLoader parameters
        dl_params = {
            "batch_size":      8,
            "num_workers":     num_cpus, # generally best if set to num of CPUs
            "prefetch_factor": 2,
            "pin_memory":      True, # if enabled uses more VRAM
            "shuffle":         True
        }

        train_dataset = pd.read_pickle("data/4th_train.pkl")
        validation_dataset = pd.read_pickle("data/4th_val.pkl")
        test_dataset = pd.read_pickle("data/4th_test.pkl")

        # Telegram status message setup
        started = datetime.datetime.now().replace(microsecond=0)
        len_all = len(train_dataset) + len(validation_dataset) + len(test_dataset)
        dl_params_str = "\n".join(f'{k}: {v}' for k, v in dl_params.items())
        send_status(f"Training started on {model_type}\n"
                    f"train dataset size: {len(train_dataset)} ({100*len(train_dataset)/len_all:.0f}%)\n"
                    f"validation dataset size: {len(validation_dataset)} ({100*len(validation_dataset)/len_all:.0f}%)\n"
                    f"test dataset size: {len(test_dataset)} ({100*len(test_dataset)/len_all:.0f}%)\n"
                    f"{dl_params_str}")

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **dl_params)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, **dl_params)
        val_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, **dl_params)

        model = T5ForConditionalGeneration.from_pretrained(model_type).to(device)

        send_status(f"Running on {torch.cuda.device_count()} GPUs and {num_cpus} CPUs")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        tokenizer = T5Tokenizer.from_pretrained(model_type)

        for epoch in range(epochs):
            # Training loop
            with tqdm(total=len(train_dataset), unit="batch") as pbar:
                for batch in train_dataloader:
                    inputs = batch["input"]
                    outputs = batch["output"]

                    input_ids = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
                    output_ids = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)

                    outputs = model(input_ids=input_ids, labels=output_ids)
                    loss = outputs.loss.sum() # or .sum() # TODO: Glušo check

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
                    val_loss += outputs.loss.sum() #! TODO: preveri, če je loss izračunan kot povprečje batcha ali je seštevek vseh samplov v batchu # TODO: Glušo check .sum()

            val_loss /= len(validation_dataset)

            epoch_status = f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}"
            print(epoch_status)
            send_status(epoch_status)

    except Exception as e:
        send_status(f"Training failed\n{e}")
        print(inputs)
        print(''.join(traceback.format_exception(None, e, e.__traceback__)))

    else:
        ok_str = "Training finished"
        finished = datetime.datetime.now().replace(microsecond=0)
        dir_name = f"{model_type.split('/')[-1]}_{finished.isoformat()[5:-3]}"
        save_dir = f"/d/hpc/projects/FRI/team9/models/{dir_name}"
        if torch.cuda.device_count() > 1: # model wrapped in DP
            model.module.save_pretrained(save_dir)
        else:
            model.save_pretrained(save_dir)
        # os.system(f"chown :fri-users -R {save_dir}")
        ok_str += f", took {finished - started}"
        send_status(ok_str)
        print(ok_str)
