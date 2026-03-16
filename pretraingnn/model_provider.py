import os
from .model import GNN_graphpred
from functools import lru_cache

CHECKPOINT_PATH = "pretraingnn/model_gin/supervised_contextpred.pth"
DEVICE = "cpu"

# Hyperparameter values as in pretrained file
NUM_LAYERS = 5
EMBEDDING_DIM = 300
NUM_TASKS = 1


@lru_cache(maxsize=1)
def get_model() -> GNN_graphpred:
    model = GNN_graphpred(
        num_layer=NUM_LAYERS,
        emb_dim=EMBEDDING_DIM,
        num_tasks=NUM_TASKS,
    )
    model.from_pretrained(CHECKPOINT_PATH, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    return model