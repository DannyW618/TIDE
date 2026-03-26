import argparse
import os
from param_utils import params
from parse import parser_add_main_args
from train_utils import build_models, build_eval_components, load_experiment_data, run_single_experiment, save_result,fix_seed,resolve_device
from logger import Logger_detect
import torch
import time
from tqdm import tqdm
from torch_sparse import SparseTensor, matmul

os.environ["CUDA_LAUNCH_BLOCKING"] = params("cuda_launch_blocking")

def main():
    parser = argparse.ArgumentParser(description="General Training Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = resolve_device(args)

    dataset_ind, dataset_ood_tr, dataset_ood_te, in_channels, \
        out_channels, decoupled = load_experiment_data(args)

    model, structure_model, feature_model = build_models(in_channels, out_channels, args, device)
    criterion, eval_func, logger = build_eval_components(args)

    struct_data_ind, struct_data_ood_te, feat_data_ind, feat_data_ood_te = decoupled

    datasets = {
        "dataset_ind": dataset_ind,
        "dataset_ood_tr": dataset_ood_tr,
        "dataset_ood_te": dataset_ood_te,
        "struct_data_ind": struct_data_ind,
        "struct_data_ood_te": struct_data_ood_te,
        "feat_data_ind": feat_data_ind,
        "feat_data_ood_te": feat_data_ood_te,
    }
    
    results = None
    for run in range(args.runs):
        logger = run_single_experiment(
            run,
            model,
            structure_model,
            feature_model,
            datasets,
            criterion,
            eval_func,
            args,
            device,
        )

        if args.train_model:
            results = logger.print_statistics(return_best_idx=True)[0]

        if args.mode == "detect" and args.train_model:
            save_result(results, args)


if __name__ == "__main__":
    main()
