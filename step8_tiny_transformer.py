
from dataclasses import dataclass

@dataclass
class ModelConfig:
    dim: int = 384          # Increased from 256
    n_layers: int = 6       # Increased from 4
    n_heads: int = 6        # dim (384) / n_heads (6) = 64 head_dim
    n_kv_heads: int = 2
    vocab_size: int = 4000  # CRITICAL: Must match your BPE tokenizer
    hidden_dim: int = None
    multiple_of: int = 32
    max_seq_len: int = 256  # Increased for better context
    norm_eps: float = 1e-5
    dropout: float = 0.1    # Added slight dropout for better generalization
