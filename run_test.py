import torch
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import os
import pandas as pd
import traceback
from calc_custom_metric import custom_score
from slurm.tg_status import send_status

def test_step(paraphraser, test_dataloader):
    with torch.no_grad():
        originals = []
        generated = []
        with tqdm(total=len(test_dataloader), unit="batch") as pbar:
            for batch in test_dataloader:
                inputs = batch["input"]
                originals.extend(inputs)
                pphrase = paraphraser(inputs)
                generated_phrases = [phrase for phrase in pphrase]
                generated.extend(generated_phrases)
                pbar.update(len(inputs))
        # print(originals[0])
        # print(generated[0])
        bleu_score = corpus_bleu([[ref] for ref in originals], generated)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        rouge_scores = scorer.score(" ".join(originals), " ".join(generated))
        # print(bleu_score)
        # print(rouge_scores)

        custom_metric_score = custom_score(generated, originals)

        return bleu_score, rouge_scores, custom_metric_score

def get_paraphraser(model_name, tokenizer_type):
    model_loc = "/d/hpc/projects/FRI/team9/models/" + model_name
    model = T5ForConditionalGeneration.from_pretrained(model_loc, local_files_only=True)
    model = model.to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_type)
    paraphraser = lambda x: [tokenizer.decode(output, skip_special_tokens=True) for output in model.generate(
            **tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
                .to("cuda")
    )]
    return paraphraser


if __name__ == "__main__":
    try:
        # Parameters
        model_name = "t5-sl-small_05-09T13:33"
        tokenizer_type = "cjvt/t5-sl-small" # original tokenizer
        num_cpus = len(os.sched_getaffinity(0))
        paraphraser = get_paraphraser(model_name, tokenizer_type)
        # DataLoader parameters
        dl_params = {
            "batch_size":      16,
            "num_workers":     num_cpus, # generally best if set to num of CPUs
            "prefetch_factor": 2,
            "pin_memory":      True, # if enabled uses more VRAM
            "shuffle":         True
        }
        test_dataset = pd.read_pickle("data/4th_test.pkl")
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, **dl_params)

        dl_params_str = "\n".join(f'{k}: {v}' for k, v in dl_params.items())
        send_status(f"Testing started on {model_name}\n"
                        f"test dataset size: {len(test_dataset)}\n"
                        f"{dl_params_str}")


        bleu_score, rouge_scores, custom_metric_score = test_step(paraphraser, test_dataloader)
    except Exception as e:
        send_status(f"Testing failed\n{e}")
        print(''.join(traceback.format_exception(None, e, e.__traceback__)))
    else:
        send_status(f"Testing completed on {model_name}\n"
                    f"test dataset size: {len(test_dataset)}\n"
                    f"{dl_params_str}"
                    f"BLEU score: {bleu_score:.4f}"
                    f"ROUGE-1 score: {rouge_scores['rouge1'].fmeasure:.4f}"
                    f"ROUGE-2 score: {rouge_scores['rouge2'].fmeasure:.4f}"
                    f"Custom metric score: {custom_metric_score:.4f}")
        print(f"Testing completed on {model_name}\n"
                    f"test dataset size: {len(test_dataset)}\n"
                    f"{dl_params_str}"
                    f"BLEU score: {bleu_score:.4f}"
                    f"ROUGE-1 score: {rouge_scores['rouge1'].fmeasure:.4f}"
                    f"ROUGE-2 score: {rouge_scores['rouge2'].fmeasure:.4f}"
                    f"Custom metric score: {custom_metric_score:.4f}")