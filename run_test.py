import torch
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from calc_custom_metric import custom_score

def test_step(model, test_dataloader, device, tokenizer):
    model.eval()

    with torch.no_grad():
        originals = []
        generated = []

        for batch in test_dataloader:
            inputs = batch["input"]
            outputs = batch["output"]

            input_ids = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
            output_ids = tokenizer(outputs, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)

            outputs = model(input_ids=input_ids, labels=output_ids)

            input_str = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            originals.extend(input_str)
            output_str = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            generated.extend(output_str)

        bleu_score = corpus_bleu([[ref] for ref in originals], generated)
        # print(f"BLEU score: {bleu_score:.4f}")

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        rouge_scores = scorer.score(originals, generated)
        # print(f"ROUGE-1 score: {rouge_scores['rouge1'].fmeasure:.4f}")
        # print(f"ROUGE-2 score: {rouge_scores['rouge2'].fmeasure:.4f}")

        custom_metric_score = custom_score(generated, originals)
        # print(f"Custom metric score: {custom_metric_score:.4f}")

        return bleu_score, rouge_scores, custom_metric_score
