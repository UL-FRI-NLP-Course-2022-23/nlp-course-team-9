"""
2. step - manual evaluation
"""

import os
import pickle as pkl
import random
from pprint import pprint

# Parameters
random.seed(1)
base_dir = 'data/back_translations/'


def merge_pkls():
    all_paragraphs = []

    for f in os.listdir(base_dir):
        f_split = f.split('.')
        if len(f_split) == 2:
            filename, extension = f_split
            if extension == 'pkl':
                with open(base_dir + f, 'rb') as pkl_file:
                    pkl_paragraphs = pkl.load(pkl_file)
                    all_paragraphs += pkl_paragraphs

    with open(base_dir + '3rd_try.pkl', 'wb') as pkl_file:
        pkl.dump(all_paragraphs, pkl_file)


if __name__ == '__main__':
    all_paragraphs = dict()
    with open(base_dir + '3rd_try.pkl', 'rb') as pkl_file:
        for i, (original, back_translation) in enumerate(pkl.load(pkl_file)):
            all_paragraphs[i] = (original, back_translation)

    print(f'{len(all_paragraphs.values())} paragraphs in total.')

    pprint(random.sample(all_paragraphs.items(), 10), width=210)
