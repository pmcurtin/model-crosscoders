default_cfg = {
    "seed": 51,
    "batch_size": 2048,
    "buffer_mult": 512,  # 512,
    "lr": 1e-3,
    "num_tokens": int(1e8),
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "dict_size": 2**15,  # some hidden dim?
    "topk": 64,
    "seq_len": 512,
    # "enc_dtype": "fp32",  # probably use fp16
    # "remove_rare_dir": False,
    # "model_name": "gpt2-small",  # not this
    # "site": "resid_post",
    # "layer": 0,
    "device": "cuda:0",  # maybee
    # "device": "cpu", # for the poor fish
    "model_batch_size": 8,
    "log_every": 100,
    "save_every": 5000,
    "dec_init_norm": 0.05,
    "save_dir": "models/some_model",
    "save_version": 0,
    # "wandb_project":
}
