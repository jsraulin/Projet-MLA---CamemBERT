class CamembertConfig:
    """Paramètres de configuration du modèle CamemBERT simplifié."""
    def __init__(self):
        self.vocab_size = 32000
        self.hidden_size = 256
        self.num_hidden_layers = 6
        self.num_attention_heads = 8
        self.intermediate_size = 1024
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.layer_norm_eps = 1e-5
        self.pad_token_id = 1
        self.hidden_act = "gelu"
