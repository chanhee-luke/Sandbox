# Sandbox
Personal NLP research platform borrowed heavily from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

## TODOs

- [ ] Refactor code to be more readable
- [ ] Add more customizable experiment layer, don't hard code everything in the future
- [ ] Diff w/ Opennmt-py, incorporate updates

## Pipeline

### Preprocess

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

### Train

```bash
python train.py -data data/demo -save_model demo-model
```

### Translate

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk
```
