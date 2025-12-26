"""RBM experiment configuration."""

config = {
    "device": "auto",  # "auto" | "cpu" | "cuda:0" | "mps"

    "data": {
        "csv_path": "data/data.csv",  # Path to training data CSV
        "drop_cols": [],
    },

    "model": {
        "bm_type": "rbm",   # Currently only RBM is supported
        "visible_blocks": {"states": 8, "returns": 505*7},
        "hidden_blocks": {"h_states": 4, "hub": 600, "h_returns": 505},
        "cross_block_restrictions": [("states", "h_returns"), ("returns", "h_states")],
        "initialization": "random",
    },

    "preprocess": {
        "q_low": 0.001,
        "q_high": 0.999,
        "add_missing_bit": False,
        "max_categories": 200,
        "min_category_freq": 2,
    },

    "dataloader": {
        "batch_size": 1000,
        "split": [0.8, 0.1, 0.1],
        "seed": 42,
        "shuffle_train": True,
        "num_workers": 0,
        "drop_last_train": True,
        "pin_memory": "auto",  # "auto" | True | False
    },

    "train": {
        "epochs": 3000,
        "lr": 1e-1,
        "k": 10,
        "kind": "mean-field",
        "momentum": 0.9,
        "weight_decay": 5e-3,
        "clip_value": 0.05,
        "clip_norm": 5.0,
        "lr_schedule": {"mode": "cosine", "min_lr": 1e-4},
        "sparse_hidden": True,
        "rho": 0.1,
        "lambda_sparse": 0.01,
        "early_stopping": False,
        "es_patience": 8,
    },

    "eval": {
        "recon_k": 1,
    },

    "conditional": {
        "clamp_idx": list(range(8)),
        "target_idx": list(range(8, 505*7)),
        "n_samples": 100,
        "burn_in": 500,
        "thin": 10,
    },
}
