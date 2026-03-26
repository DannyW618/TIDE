# Credit Goes to: https://github.com/Linear95/CLUB/blob/master/mi_estimators.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from param_utils import params, ratio

class CLUB(nn.Module):
    def __init__(self, hidden_size):
        super(CLUB, self).__init__()
        x_dim = y_dim = hidden_size
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // params("info_hidden_divisor")),
            nn.ReLU(),
            nn.Linear(hidden_size // params("info_hidden_divisor"), y_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // params("info_hidden_divisor")),
            nn.ReLU(),
            nn.Linear(hidden_size // params("info_hidden_divisor"), y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        normalizer = 1 / ratio("gaussian_half")
        positive = -((mu - y_samples) ** 2) / normalizer / logvar.exp()
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / normalizer / logvar.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)