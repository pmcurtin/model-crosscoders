default_cfg = {
    "seed": 51,
    "batch_size": 2048,
    "buffer_mult": 512,
    "lr": 2e-5,
    "num_tokens": int(4e8),
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "dict_size": 2**16, # some hidden dim?
    "seq_len": 1024,
    "enc_dtype": "fp32", # probably use fp16
    # "remove_rare_dir": False,
    "model_name": "gpt2-small", # not this
    "site": "resid_post",
    # "layer": 0,
    "device": "cuda:0", # maybee
    "model_batch_size": 32,
    "log_every": 100,
    "save_every": 100000,
    "dec_init_norm": 0.005,
    
    "modelA_resid_size": None, # PLACEHOLDER; HERE UNTIL WE FIGURE OUT HOW TO GET THE ACTUAL SIZES
    "modelB_resid_size": None, # PLACEHOLDER; HERE UNTIL WE FIGURE OUT HOW TO GET THE ACTUAL SIZES

}