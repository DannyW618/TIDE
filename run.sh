# Cora structure
python main.py --method tide --backbone gcnib --dataset cora --ood_type structure --mode detect --reset --use_bn --use_prop --device 0 --beta 0.01 --dropout 0.5 --lr 0.001 --gamma 1 --train_model --train_structure --train_feature --use_pairwise --pmi_w 0.01

# Cora feature
python main.py --method tide --backbone gcnib --dataset cora --ood_type feature --mode detect --reset --use_bn --use_prop --device 0 --beta 0.01 --dropout 0.5 --lr 0.001 --gamma 0.1 --train_model --use_pairwise --train_feature --train_structure --pmi_w 0.01

# Pubmed structure
python main.py --method tide --backbone gcnib --dataset pubmed --ood_type structure --mode detect --reset --use_bn --use_prop --device 0 --beta 0.005 --dropout 0.5 --lr 0.001 --train_model --gamma 1 --use_pairwise --train_feature --train_structure --pmi_w 0.01

# Pubmed feature
python main.py --method tide --backbone gcnib --dataset pubmed --ood_type feature --mode detect --reset --use_bn --use_prop --device 0 --beta 0.005 --dropout 0.5 --lr 0.001 --train_model --gamma 1 --use_pairwise --train_feature --train_structure --pmi_w 0.01

# Twitch
python main.py --method tide --backbone gcnib --dataset twitch --mode detect --use_bn --use_prop --device 0 --train_model --lr 0.0001 --dropout 0 --T 0.1 --beta 0.0001 --gamma 0.02 --use_pairwise --train_feature --train_structure --pmi_w 0.0001

# Citeseer Label
python main.py --method tide --backbone gcnib --dataset citeseer --ood_type label --mode detect --reset --use_bn --use_prop --train_model --device 0 --beta 0.0001 --dropout 0 --lr 0.001 --gamma 0.5 --use_pairwise --train_feature --train_structure --pmi_w 0.0001
