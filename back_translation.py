import gc
from nemo.collections.nlp.models import MTEncDecModel
import os
import pickle as pkl
from pprint import pprint
import re
# from slurm.tg_status import send_status
import torch

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# general parameters
min_sentence_len = 10
min_lines_for_paragraph = 2


# send_status("loading models")
model_slen = MTEncDecModel.restore_from('models/v1.2.6/slen/aayn_base.nemo').cuda()
model_ensl = MTEncDecModel.restore_from('models/v1.2.6/ensl/aayn_base.nemo').cuda()

print("reading files")
files_in_dir = sorted(os.listdir("data/cckresV1_0-text"))
# files_in_dir = ["F0000246.txt"] # TESTING ONLY

for i, f in enumerate(files_in_dir):
    filename, _ = f.split(".")
    # if i % 1000 == 0:
    #     send_status(f"{i + 1}/{len(files_in_dir)} ({filename})")

    with open("data/cckresV1_0-text/" + f) as sl_file:
        sl = "\n".join(line.strip() for line in sl_file.readlines())

    pattern = r"((?:.{" + str(min_sentence_len) + r",}\n){" + str(min_lines_for_paragraph) + r",})\n"
    sl_paragraphs = []
    for group in re.findall(pattern, sl):
        if len(group) > 512:
            while len(group.split("\n")[0]) < 512: # to avoid "chuck too big" errors, TODO: use leftover group
                group = group.replace("\n", " ", 1)
        sl_paragraphs.append(group.split("\n")[0])

    if len(sl_paragraphs) > 0:
        bt_paragraphs = []

        print(f"translating {filename}")

        for sl_paragraph in sl_paragraphs:
            sl_en = model_slen.translate([sl_paragraph.strip()])
            sl_en_sl = model_ensl.translate(sl_en)
            bt_paragraphs += sl_en_sl

            # print(sl_paragraph + "\n" + sl_en_sl[0])

        ok_paraphrases = []

        # filter out non-paraphrased
        for s, ses in zip(sl_paragraphs, bt_paragraphs):
            if s != ses and len(s) > 10 and len(ses) > 10:
                ok_paraphrases.append((s, ses))

        # pprint(ok_paraphrases)

        with open(f"data/back_translations/{filename}.pkl", "wb") as pkl_file:
            pkl.dump(ok_paraphrases, pkl_file) # write list of tuples
            print(f"{filename}.pkl written")

    else:
        print(f"{filename} contains no usable text")

# tips from labs:
# - parsent bitext mining
# - margin based parallel corpus mining
