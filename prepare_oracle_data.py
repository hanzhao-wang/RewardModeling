from __future__ import annotations

from typing import *

import click
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def infer_rm_score_formatted(
    ds: Dataset,
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B",
    save_name: str = "dummy",
):
    ds_processed = []

    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # specified by official
        num_labels=1,
    )

    dummy_device = "cuda:0"

    for i in tqdm(range(len(ds)), desc=f"RM inference with {model_name}"):
        pos_input = rm_tokenizer.apply_chat_template(
            ds[i]["chosen"], tokenize=True, return_tensors="pt"
        ).to(dummy_device)
        neg_input = rm_tokenizer.apply_chat_template(
            ds[i]["rejected"], tokenize=True, return_tensors="pt"
        ).to(dummy_device)

        with torch.no_grad():
            pos_out = rm(pos_input).logits[0][0].item()
            neg_out = rm(neg_input).logits[0][0].item()

        sample = ds[i]
        sample.update(
            {
                "chosen_score": pos_out,
                "rejected_score": neg_out,
            }
        )
        ds_processed.append(sample)

    ds_rslt = Dataset.from_list(ds_processed)
    ds_rslt.save_to_disk(f"statdata/{save_name}_{model_name}")


@click.command()
@click.option(
    "--built_from", type=click.Choice(["local", "hub"]), default="local"
)
@click.option(
    "--data_path",
    type=str,
    default="RLHFlow/PKU-SafeRLHF-30K-standard",
)
def main(built_from: Literal["local", "hub"],
data_path: str = "RLHFlow/UltraFeedback-preference-standard"):
    if built_from == "local":
        ds = load_dataset(
            data_path, name='default', split="test"
        )
        '''
        infer_rm_score_formatted(
            ds,
            model_name="Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
            save_name="",
        )
        '''
        infer_rm_score_formatted(
            ds,
            model_name="PKU-Alignment/beaver-7b-v2.0-reward",
            save_name="",
        )
        
    elif built_from == "hub":
        if data_path == "Skywork-Reward-Preference-80K-v0.2":
            ds = load_dataset(
                "BigCatc/Skywork-Reward-Preference-80K-v0.2-ordinal", split="train"
            )
            ds.save_to_disk(
                "statdata/prefer_skywork_Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
            )
        else:
            ds = load_dataset(
                data_path, split="train"
            )
            ds.save_to_disk(
                "statdata/"+data_path
            )



if __name__ == "__main__":
    main()
