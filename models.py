"""
- CSC 580 - Artificial Intelligence II, Winter 2026, Section 801
- Default Final Project - Option 1 (World Model: Dreamer V1 World Model Components for Highway-env)
- Student name: Amarsaikhan Batjargal
- AI Tools Consulted: Claude (Anthropic) for code structure guidance and debugging  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Default dimensions
IMG_CHANNELS = 3
IMG_HEIGHT = 64
IMG_WIDTH = 64
LATENT_DIM = 32
DETERMINISTIC_DIM = 200
HIDDEN_DIM = 200
ENCODER_DEPTH = 32
ACTION_DIM = 2


class Encoder(nn.Module):
    """CNN Encoder: maps RGB frames to a latent embedding."""

    def __init__(self, img_channels=IMG_CHANNELS, depth=ENCODER_DEPTH,
                 latent_dim=LATENT_DIM):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(img_channels, 1 * depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.feature_dim = 8 * depth * 4 * 4
        self.fc = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, obs):
        x = self.convs(obs)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    """Transposed CNN Decoder: reconstructs frames from latent states."""

    def __init__(self, state_dim=DETERMINISTIC_DIM + LATENT_DIM,
                 depth=ENCODER_DEPTH, img_channels=IMG_CHANNELS):
        super().__init__()
        self.feature_dim = 8 * depth * 4 * 4
        self.depth = depth
        self.fc = nn.Linear(state_dim, self.feature_dim)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(8 * depth, 4 * depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * depth, 1 * depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1 * depth, img_channels, 4, stride=2, padding=1),
        )

    def forward(self, state):
        x = self.fc(state)
        x = x.view(-1, 8 * self.depth, 4, 4)
        return self.deconvs(x)


class RSSM(nn.Module):
    """Recurrent State-Space Model: core of Dreamer's world model."""

    def __init__(self, latent_dim=LATENT_DIM, deterministic_dim=DETERMINISTIC_DIM,
                 hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.deterministic_dim = deterministic_dim

        self.pre_gru = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deterministic_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def initial_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return {
            'deterministic': torch.zeros(batch_size, self.deterministic_dim, device=device),
            'stochastic': torch.zeros(batch_size, self.latent_dim, device=device),
        }

    def observe_step(self, prev_state, action, encoded_obs):
        x = self.pre_gru(torch.cat([prev_state['stochastic'], action], dim=-1))
        deterministic = self.gru(x, prev_state['deterministic'])

        prior_params = self.prior_net(deterministic)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1
        prior_dist = Normal(prior_mean, prior_std)

        posterior_input = torch.cat([deterministic, encoded_obs], dim=-1)
        posterior_params = self.posterior_net(posterior_input)
        posterior_mean, posterior_log_std = posterior_params.chunk(2, dim=-1)
        posterior_std = F.softplus(posterior_log_std) + 0.1
        posterior_dist = Normal(posterior_mean, posterior_std)

        stochastic = posterior_dist.rsample()

        new_state = {
            'deterministic': deterministic,
            'stochastic': stochastic,
        }
        return new_state, prior_dist, posterior_dist

    def imagine_step(self, prev_state, action):
        x = self.pre_gru(torch.cat([prev_state['stochastic'], action], dim=-1))
        deterministic = self.gru(x, prev_state['deterministic'])

        prior_params = self.prior_net(deterministic)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1
        prior_dist = Normal(prior_mean, prior_std)

        stochastic = prior_dist.rsample()

        new_state = {
            'deterministic': deterministic,
            'stochastic': stochastic,
        }
        return new_state, prior_dist

    def get_full_state(self, state):
        return torch.cat([state['deterministic'], state['stochastic']], dim=-1)


class RewardModel(nn.Module):
    """MLP that predicts scalar rewards from the combined latent state."""

    def __init__(self, state_dim=DETERMINISTIC_DIM + LATENT_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class Actor(nn.Module):
    """Policy network outputting continuous actions."""

    def __init__(self, state_dim=DETERMINISTIC_DIM + LATENT_DIM,
                 hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_action(self, state, deterministic=False):
        dist = self.forward(state)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()
        return torch.tanh(raw_action)


class Critic(nn.Module):
    """Value network estimating expected returns."""

    def __init__(self, state_dim=DETERMINISTIC_DIM + LATENT_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def load_checkpoint(path, device=None):
    """Load all models from a training checkpoint.

    Args:
        path: Path to the .pt checkpoint file
        device: torch device (auto-detects if None)

    Returns:
        Dict with keys: encoder, rssm, reward_model, decoder, actor, critic
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    models = {
        'encoder': Encoder().to(device),
        'rssm': RSSM().to(device),
        'reward_model': RewardModel().to(device),
        'decoder': Decoder().to(device),
        'actor': Actor().to(device),
        'critic': Critic().to(device),
    }

    for name, model in models.items():
        model.load_state_dict(checkpoint[name])
        model.eval()

    return models