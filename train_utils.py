import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from param_utils import params, ratio
from contrastive_loss import CLUB
from data_utils import (
    decouple_data,
    eval_acc,
    eval_rocauc,
    evaluate_detect,
    mutual_independence_loss,
    rand_splits,
    train_step,
)
from dataset import load_dataset
from logger import Logger_classify, Logger_detect, save_result
from tide import TIDE

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_2d_labels(data):
    if isinstance(data, list):
        for datum in data:
            if len(datum.y.shape) == 1:
                datum.y = datum.y.unsqueeze(1)
        return

    if len(data.y.shape) == 1:
        data.y = data.y.unsqueeze(1)


def resolve_device(args):
    return torch.device(
        "cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )


def load_experiment_data(args):
    dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

    ensure_2d_labels(dataset_ind)
    ensure_2d_labels(dataset_ood_tr)
    ensure_2d_labels(dataset_ood_te)

    if args.dataset not in ["cora", "citeseer", "pubmed"]:
        dataset_ind.splits = rand_splits(
            dataset_ind.node_idx,
            train_prop=args.train_prop,
            valid_prop=args.valid_prop,
        )

    print(dataset_ind, dataset_ood_te)

    out_channels = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
    in_channels = dataset_ind.x.shape[1]
    decoupled = decouple_data(dataset_ind, dataset_ood_te)

    return dataset_ind, dataset_ood_tr, dataset_ood_te, in_channels, out_channels, decoupled


def build_models(in_channels, out_channels, args, device):
    model = TIDE(in_channels, out_channels, args).to(device)

    struct_args = copy.deepcopy(args)
    structure_model = TIDE(in_channels, out_channels, struct_args).to(device)

    feat_args = copy.deepcopy(args)
    feat_args.backbone = "mlp"
    feature_model = TIDE(in_channels, out_channels, feat_args).to(device)

    return model, structure_model, feature_model


def build_eval_components(args):
    criterion = nn.NLLLoss()
    eval_func = eval_rocauc if args.dataset in ("proteins", "ppi", "twitch") else eval_acc
    logger = Logger_classify(args.runs, args) if args.mode == "classify" else Logger_detect(args.runs, args)
    return criterion, eval_func, logger


def reset_run_models(model, structure_model, feature_model, args, device):
    if args.reset:
        model.reset_parameters()
        structure_model.reset_parameters()
        feature_model.reset_parameters()

    model.to(device)
    structure_model.to(device)
    feature_model.to(device)


def build_pairwise_modules(hidden_channels, device):
    club_zq = CLUB(hidden_channels).to(device)
    club_zv = CLUB(hidden_channels).to(device)
    club_qv = CLUB(hidden_channels).to(device)
    return club_zq, club_zv, club_qv


def build_optimizers(args, model, structure_model, feature_model, club_zq, club_zv, club_qv):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    structure_optimizer = torch.optim.Adam(structure_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    feature_optimizer = torch.optim.Adam(feature_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    club_zv_optimizer = torch.optim.Adam(
        list(club_zv.parameters()) + list(structure_model.parameters()) + list(model.parameters()),
        lr=ratio("club_lr"),
    )
    club_zq_optimizer = torch.optim.Adam(
        list(club_zq.parameters()) + list(feature_model.parameters()) + list(model.parameters()),
        lr=ratio("club_lr"),
    )
    club_qv_optimizer = torch.optim.Adam(
        list(club_qv.parameters()) + list(structure_model.parameters()) + list(feature_model.parameters()),
        lr=ratio("club_lr"),
    )

    return {
        "main": optimizer,
        "structure": structure_optimizer,
        "feature": feature_optimizer,
        "club_zv": club_zv_optimizer,
        "club_zq": club_zq_optimizer,
        "club_qv": club_qv_optimizer,
    }

def run_training_epoch(
    model,
    structure_model,
    feature_model,
    datasets,
    criterion,
    args,
    device,
    clubs,
    optimizers,
    epoch,
):
    dataset_ind, dataset_ood_tr = datasets["dataset_ind"], datasets["dataset_ood_tr"]
    struct_data_ind, feat_data_ind = datasets["struct_data_ind"], datasets["feat_data_ind"]

    club_zq, club_zv, club_qv = clubs

    model.train()

    if args.use_pairwise:
        club_zq.train()
        club_zv.train()
        club_qv.train()

    if args.train_structure:
        structure_model.train()
        structure_loss, _ = structure_model.loss_compute(
            struct_data_ind, dataset_ood_tr, criterion, device, args, decoupled=1
        )
        optimizers["structure"] = train_step(optimizers["structure"], structure_loss)

    if args.train_feature:
        feature_model.train()
        feature_loss, _ = feature_model.loss_compute(
            feat_data_ind, dataset_ood_tr, criterion, device, args, decoupled=1
        )
        optimizers["feature"] = train_step(optimizers["feature"], feature_loss)
    if args.use_pairwise and epoch % params("interval") == 0:
        optimizers["club_qv"] = train_step(
            optimizers["club_qv"],
            mutual_independence_loss(
                structure_model,
                feature_model,
                struct_data_ind,
                feat_data_ind,
                club_qv,
                device,
                weight=args.pmi_w,
                name="qv",
            ),
        )
        optimizers["club_zq"] = train_step(
            optimizers["club_zq"],
            mutual_independence_loss(
                feature_model,
                model,
                feat_data_ind,
                dataset_ind,
                club_zq,
                device,
                weight=args.pmi_w,
                name="zq",
            ),
        )
        optimizers["club_zv"] = train_step(
            optimizers["club_zv"],
            mutual_independence_loss(
                structure_model,
                model,
                struct_data_ind,
                dataset_ind,
                club_zv,
                device,
                weight=args.pmi_w,
                name="zv",
            ),
        )

    if not args.train_model:
        return None

    loss, _ = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
    optimizers["main"] = train_step(optimizers["main"], loss)
    return loss


def evaluate_training_run(
    model,
    structure_model,
    feature_model,
    datasets,
    criterion,
    eval_func,
    args,
    clubs,
    device,
    best_val,
):
    club_zq, club_zv, club_qv = clubs

    return evaluate_detect(
        model,
        structure_model,
        feature_model,
        datasets["dataset_ind"],
        datasets["dataset_ood_te"],
        datasets["struct_data_ind"],
        datasets["struct_data_ood_te"],
        datasets["feat_data_ind"],
        datasets["feat_data_ood_te"],
        criterion,
        eval_func,
        args,
        club_zq,
        club_zv,
        club_qv,
        device,
        best_val=best_val,
    )


def run_single_experiment(
    run,
    model,
    structure_model,
    feature_model,
    datasets,
    criterion,
    eval_func,
    args,
    device,
):
    reset_run_models(model, structure_model, feature_model, args, device)
    best_val = float("inf")

    logger = Logger_detect(args.runs, args)

    clubs = build_pairwise_modules(args.hidden_channels, device)
    optimizers = build_optimizers(args, model, structure_model, feature_model, *clubs)

    start = time.time()
    for epoch in tqdm(range(args.epochs)):
        run_training_epoch(
            model,
            structure_model,
            feature_model,
            datasets,
            criterion,
            args,
            device,
            clubs,
            optimizers,
            epoch,
        )

        if args.train_model:
            result, best_val, _ = evaluate_training_run(
                model,
                structure_model,
                feature_model,
                datasets,
                criterion,
                eval_func,
                args,
                clubs,
                device,
                best_val,
            )
            logger.add_result(run, result)

    print("Time Completed:", time.time() - start)
    return logger
