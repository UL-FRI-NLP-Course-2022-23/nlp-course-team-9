import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# Define a function to calculate the custom score
def custom_score(originals, paraphrases):
    # Load Slovenian BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
    model = AutoModel.from_pretrained("EMBEDDIA/sloberta").to("cuda")

    custom_score = 0.0
    with tqdm(total=len(originals), unit="example") as pbar:
        for original, paraphrase in zip(originals, paraphrases):
            # Tokenize both original and paraphrase
            original_tokens = word_tokenize(original.lower())
            paraphrase_tokens = word_tokenize(paraphrase.lower())

            # Count the n-grams in both original and paraphrase
            n = 2
            original_ngrams = Counter(zip(*[original_tokens[i:] for i in range(n)]))
            paraphrase_ngrams = Counter(zip(*[paraphrase_tokens[i:] for i in range(n)]))

            # Calculate the n-gram overlap
            overlap = sum((original_ngrams & paraphrase_ngrams).values())
            
            # Calculate the length difference
            length_diff = abs(len(original_tokens) - len(paraphrase_tokens))

            # Calculate the semantic similarity
            original_emb = model(tokenizer.encode(original, padding=True, truncation=True, max_length=512, return_tensors='pt').to("cuda"))[0][0]
            paraphrase_emb = model(tokenizer.encode(paraphrase, padding=True, truncation=True, max_length=512, return_tensors='pt').to("cuda"))[0][0]

            original_emb_avg = torch.mean(original_emb, dim=0, keepdim=True)
            paraphrase_emb_avg = torch.mean(paraphrase_emb, dim=0, keepdim=True)

            cosine_similarity_score = cosine_similarity(original_emb_avg.detach().numpy().reshape(1, -1), paraphrase_emb_avg.detach().numpy().reshape(1, -1))[0][0]

            original_set = set(original_tokens)
            paraphrase_set = set(paraphrase_tokens)

            jaccard_similarity = len(original_set.intersection(paraphrase_set)) / len(original_set.union(paraphrase_set))

            # Calculate the combined similarity score
            combined_similarity_score = (cosine_similarity_score + jaccard_similarity) / 2.0

            # Calculate the custom score by combining all the metrics
            custom_score += 0.25 * overlap / len(original_tokens) + 0.25 * (1 - length_diff / len(original_tokens)) + 0.5 * combined_similarity_score
            
            pbar.update(1)

    return custom_score / len(originals)


if __name__ == "__main__":
    original ="Ko je dvojna monarhija propadla, ni Vauhnik omahoval niti za hip, ampak se je takoj pridružil skupini oficirjev in vojakov okoli generala Maistra. Bil je med tistimi, ki jim je uspelo obraniti Maribor in je v letu 1919 sodeloval v bojih na Koroškem. Vauhnik je bil ranjen, je pa tudi napredoval in postal stotnik. Ob koncu bojev na Koroškem, ko je vprašanje meja iz rok vojakov prevzela diplomacija, je Vauhnik zaprosil za sprejem v vojsko Kraljevine Srbov, Hrvatov in Slovencev. Sprejeli so ga in mu potrdili stotniški čin."
    
    paraphrase="Po razpadu dvojne monarhije se Vauhnik ni niti za trenutek spotaknil, ampak se je takoj pridružil skupini častnikov in vojakov okoli generala Maistra. Bil je med tistimi, ki so uspeli braniti Maribor in leta 1919 sodelovali v bojih na Koroškem. Vauhnik je bil ranjen, vendar je tudi napredoval in postal kapetan. Ob koncu bitk na Koroškem, ko je vprašanje meje prevzela diplomacija, je Vauhnik zaprosil za sprejem v vojsko Kraljevine Hrvatov, Srbov in Slovencev."
    print(custom_score([original], [paraphrase]))

