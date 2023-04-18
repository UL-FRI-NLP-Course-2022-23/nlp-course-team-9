from nemo.collections.nlp.models import MTEncDecModel
import os
import pickle as pkl
from pprint import pprint
import re


model_slen = MTEncDecModel.restore_from('models/v1.2.6/slen/aayn_base.nemo')
model_ensl = MTEncDecModel.restore_from('models/v1.2.6/ensl/aayn_base.nemo')

for f in sorted(os.listdir("data/cckresV1_0-text")):
    filename, _ = f.split(".")
    print(filename)

    with open("data/cckresV1_0-text/" + f) as sl_file:
        sl = "\n".join(line.strip() for line in sl_file.readlines())

    pattern = r"((?:.+\n){2,})\n"
    sl_paragraphs = []
    for group in re.findall(pattern, sl):
        if len(group) < 512: # to avoid "chuck too big" errors
            group = group.replace("\n", " ")
            sl_paragraphs.append(group)

    if len(sl_paragraphs) == 0:
        continue

    sl_en = model_slen.translate(sl_paragraphs)
    sl_en_sl = model_ensl.translate(sl_en)

    ok_paraphrases = []

    # filter out non-paraphrased
    for s, ses in zip(sl_paragraphs, sl_en_sl):
        if s != ses:    # TODO: add some threshold eg. Levenshtein distance
            ok_paraphrases.append((s, ses))

    # pprint(ok_paraphrases)

    with open(f"data/back_translations/{filename}.pkl", "wb") as pkl_file:
        pkl.dump(ok_paraphrases, pkl_file) # write list of tuples

    # break # for testing TODO: remove

# tips from labs:
# - parsent bitext mining
# - margin based paralled corpus mining
