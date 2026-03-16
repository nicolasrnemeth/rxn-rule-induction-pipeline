import torch
import torch.nn as nn
from typing import Tuple


class PotentialFunction(nn.Module):
    """
    Fθ : R^F → R
    Matches the lightweight Agent-style MLP: feat_dim → hidden → 1.
    Higher potential ⇒ lower probability via φ(e) = σ(-Fθ(f(e))).
    """
    def __init__(self, feat_dim: int, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, edge_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        edge_feats: (E, F)
        returns:
            potentials: (E,)
            value: (1,) scalar tensor representing state value
        """
        # Policy: score each edge
        potentials = self.policy_net(edge_feats).squeeze(-1)
        
        # Value: aggregate edge features to get graph-level representation
        # Simple mean pooling
        graph_rep = edge_feats.mean(dim=0, keepdim=True) # (1, F)
        value = self.value_net(graph_rep).squeeze(-1) # (1,)
        
        return potentials, value
    
    @staticmethod
    def potentials_to_phi(potentials: torch.Tensor) -> torch.Tensor:
        """
        φ(e) = σ(-Fθ(f(e)))  (element-wise)
        """
        return torch.sigmoid(-potentials)

    def forward_with_phi(self, edge_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        potentials, value = self.forward(edge_feats)
        phi = self.potentials_to_phi(potentials)
        return potentials, phi, value