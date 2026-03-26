# TIDE

This repository contains the implementation TIDE. Sample results are shown in ```./results```

## Dependencies
- Python 3.9.19
- Cuda 12.2
- torch 2.3.1
- ogb 1.3.3
- torch_geometric 2.0.3
- torch_sparse 0.6.18

## Usage
The dataset and splits utilised are downloaded automatically when running the training scripts. Alternatively, the datasets can be downloaded via running the ```load_dataset``` function in the ```dataset.py``` file.

To execute sample runs, run the ``` run.sh ``` file.
```shell
./run.sh
```
Alternatively, you could also run the following command for individual datasets (i.e., Cora-structure):
```shell
python main.py --method tide --backbone gcnib --dataset cora --ood_type structure --mode detect --reset --use_bn --use_prop --device 0 --beta 0.01 --dropout 0.5 --lr 0.001 --gamma 1 --train_model --train_structure --train_feature --use_pairwise --pmi_w 0.01
```
