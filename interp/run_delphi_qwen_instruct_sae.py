from delphi_sae import delphi_autointerp


# model_path = "../models/qwen_instruct_sae/version_0/"
# name = "9"
# layer = 18

# model_str = "Qwen/Qwen2.5-0.5B-Instruct"
# model_dim = 896

# run_name = "sae"

if __name__ == "__main__":
    delphi_autointerp(
        model_path="../models/qwen_instruct_sae/version_0/",
        checkpoint="9",
        model_str="Qwen/Qwen2.5-0.5B-Instruct",
        model_dim=896,
        layer=18,
        run_name="sae",
    )
