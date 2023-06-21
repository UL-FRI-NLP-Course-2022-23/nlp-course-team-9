# Natural language processing course 2022/23: Sentence paraphrasing

Team members:
 * `Nik Pirnat`, `np2057@student.uni-lj.si`
 * `Martin Bavčar`, `mb9531@student.uni-lj.si`
 * `Anže Glušič`, `ag5072@student.uni-lj.si`

Group public acronym/name: TM9


## Enviroment setup

```bash
conda create -n nlp-project python=3.8 -c conda-forge
conda activate nlp-project
pip install -r requirements.txt
```

## Preprocessing

Preprocessing the ccKres dataset was done with `preprocessing.py`.


## Back translation

Back-translated dataset was computed using Slovene NMT model with `back_translation.py`.


## Training

Training was run using `run_train.py` and `run_test.py`.


## Inference

Our models can be downloaded [here](https://drive.google.com/file/d/1LL9RXX8AdQGGl_fltzEeC_38HRzW6sbp/view?usp=sharing). Refer to `inference.ipynb` to run inference on t5-sl-large and t5-sl-small models. Refer to `baseline.ipynb` for baseline.
