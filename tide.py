# Credit goes to https://github.com/qitianwu/GraphOOD-GNNSafe
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from backbone import *

class TIDE(nn.Module):
    def __init__(self, d, c, args):
        super().__init__()

        if args.backbone == "gcn":
            self.encoder = GCN(
                in_channels=d,
                hidden_channels=args.hidden_channels,
                out_channels=c,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=args.use_bn,
            )
        elif args.backbone == "mlp":
            self.encoder = MLP(
                in_channels=d,
                hidden_channels=args.hidden_channels,
                out_channels=c,
                num_layers=args.num_layers,
                dropout=args.dropout,
                std_w=args.std_w,
            )
        elif args.backbone == "gat":
            self.encoder = GAT(
                d,
                args.hidden_channels,
                c,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=args.use_bn,
            )
        elif args.backbone == "mixhop":
            self.encoder = MixHop(
                d,
                args.hidden_channels,
                c,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.backbone == "gcnjk":
            self.encoder = GCNJK(
                d,
                args.hidden_channels,
                c,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.backbone == "gatjk":
            self.encoder = GATJK(
                d,
                args.hidden_channels,
                c,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.backbone == "gcnib":
            self.encoder = GCNIB(
                in_channels=d,
                hidden_channels=args.hidden_channels,
                out_channels=c,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=args.use_bn,
                std_w=args.std_w,
            )
        else:
            raise NotImplementedError(f"Unsupported backbone: {args.backbone}")

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        """Return predicted logits."""
        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, eta=0.5):
        """Energy belief propagation. Return propagated energy."""
        e = e.unsqueeze(1)
        num_nodes = e.shape[0]

        row, col = edge_index
        deg = degree(col, num_nodes).float()
        deg_inv = 1.0 / deg[col]

        value = torch.ones_like(row) * deg_inv
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        adj = SparseTensor(
            row=col,
            col=row,
            value=value,
            sparse_sizes=(num_nodes, num_nodes),
        )

        for _ in range(prop_layers):
            e = e * eta + matmul(adj, e) * (1 - eta)

        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):
        """Return negative energy for the selected nodes."""
        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device)

        if args.backbone == "gcn":
            logits, _ = self.encoder(x, edge_index)
        else:
            logits, _, _, _, _, _, _ = self.encoder(x, edge_index)

        neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)

        if args.use_prop:
            neg_energy = self.propagation(neg_energy, edge_index, args.K, args.eta)

        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args, decoupled=None):
        """Return training loss."""
        x_in = dataset_ind.x.to(device)
        edge_index_in = dataset_ind.edge_index.to(device)
        x_out = dataset_ood.x.to(device)
        edge_index_out = dataset_ood.edge_index.to(device)

        train_in_idx = dataset_ind.splits["train"]
        train_ood_idx = dataset_ood.node_idx

        kl_loss = 0
        recon_loss = torch.tensor(0)

        if args.backbone == "gcn":
            logits_in, embed_in = self.encoder(x_in, edge_index_in)
            logits_out, _ = self.encoder(x_out, edge_index_out)
        else:
            logits_in, embed_in, mu, std, x_recon_in, mu_q, log_var_q = self.encoder(x_in, edge_index_in)
            outs = self.encoder(x_out, edge_index_out)
            logits_out = outs[0]

            if args.beta != 0:
                kl_loss = (-0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2)))

            if (decoupled is None) and (args.gamma != 0):
                recon_loss = F.l1_loss(x_recon_in[train_in_idx], x_in[train_in_idx])
                log_q = -0.5 * torch.mean(
                    log_var_q + (x_in - mu_q) ** 2 / torch.exp(log_var_q)
                )
                recon_loss = recon_loss - 0.01 * log_q

        pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
        sup_loss = criterion(
            pred_in,
            dataset_ind.y[train_in_idx].squeeze(1).to(device),
        )

        if args.use_reg and (decoupled is None):
            energy_in = -args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            energy_out = -args.T * torch.logsumexp(logits_out / args.T, dim=-1)

            if args.use_prop:
                energy_in = self.propagation(
                    energy_in, edge_index_in, args.K, args.eta
                )[train_in_idx]
                energy_out = self.propagation(
                    energy_out, edge_index_out, args.K, args.eta
                )[train_ood_idx]
            else:
                energy_in = energy_in[train_in_idx]
                energy_out = energy_out[train_ood_idx]

            # Safe check
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            reg_loss = torch.mean(
                F.relu(energy_in - args.m_in) ** 2
                + F.relu(args.m_out - energy_out) ** 2
            )

            loss = sup_loss + args.alpha * reg_loss
        else:
            loss = sup_loss

        loss = loss + args.gamma * recon_loss + args.beta * kl_loss

        return loss, embed_in.detach()
