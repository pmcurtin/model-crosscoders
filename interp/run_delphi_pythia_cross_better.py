from delphi_cross import delphi_autointerp

if __name__ == "__main__":
    delphi_autointerp(
        model_path="../models/pythia_crosscoder_better/version_1",
        checkpoint="9",
        better=True,
        model_a_str="EleutherAI/pythia-160m",
        model_dim_a=768,
        model_b_str="EleutherAI/pythia-410m-v0",
        model_dim_b=1024,
        right=True,
        layer=23,
        run_name="pythia_cross_pile_better",
    )
