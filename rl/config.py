from dataclasses import dataclass

@dataclass
class PotentialRLConfig:
    max_episodes: int = 150
    max_steps_per_episode: int = 40
    
    # Optimization
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Potential model architecture
    hidden_size: int = 128
    dropout: float = 0.1
    
    # RL specifics
    entropy_coef: float = 0.02
    baseline_momentum: float = 0.9  
    frequency_bias_alpha: float = 1.0  
    
    whitening_epsilon: float = 1e-6
    log_interval: int = 5
    
    # Device & reproducibility
    device: str = "cpu"
    seed: int = 42