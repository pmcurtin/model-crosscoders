from delphi_cross import delphi_autointerp

if __name__ == "__main__":
    delphi_autointerp(
        model_path="../models/qwen_crosscoder_better/version_3",
        checkpoint="9",
        better=True,
        model_a_str="Qwen/Qwen2.5-0.5B",
        model_dim_a=896,
        model_b_str="Qwen/Qwen2.5-0.5B-Instruct",
        model_dim_b=896,
        right=True,
        layer=18,
        run_name="qwen_cross_pile_better",
    )
