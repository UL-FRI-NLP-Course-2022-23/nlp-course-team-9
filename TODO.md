# TODO

- ročno evalvirat podatke iz back-translationa
- razdelit na train/validation/test set (to samo nardiš nek split random števil glede na število vrstic dataframa in te indekse pošlješ kot parameter, ko se kreira MyDataset in samo "slice"-aš ta dateframe.)
- validirat po vsakem epochu
- na koncu pognati na testnih podatkih
- izbrati kako velik model uporabiti ali pa kakšnega drugega (https://www.kdnuggets.com/2023/02/simple-nlp-pipelines-huggingface-transformers.html)
- shraniti natreniran model na koncu
- mogoče mal povečat learning rate pa dodat nek weight_decay (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
