"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --behaviors sycophancy
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def debug_token_positions(tokenizer, tokens, model_name="debug"):
    """
    Debug function to see what tokens are at different positions
    This helps you understand which position to extract activations from
    """
    print(f"\n=== Debug Token Positions for {model_name} ===")
    
    if len(tokens.shape) > 1:
        tokens = tokens[0]  # Remove batch dimension if present
    
    tokens_list = tokens.tolist()
    print(f"Total tokens: {len(tokens_list)}")
    print(f"Last 5 tokens: {tokens_list[-5:]}")
    
    # Decode tokens individually
    for i, token_id in enumerate(tokens_list[-5:], start=len(tokens_list)-5):
        try:
            token_text = tokenizer.decode([token_id])
            print(f"Position {i} (from end: {i - len(tokens_list)}): {token_id} -> '{token_text}'")
        except:
            print(f"Position {i} (from end: {i - len(tokens_list)}): {token_id} -> <DECODE_ERROR>")
    
    # Full sequence for context
    try:
        full_text = tokenizer.decode(tokens_list)
        print(f"\nFull sequence:\n'{full_text}'")
    except:
        print("\nCouldn't decode full sequence")
    
    print("="*50)

class ComparisonDataset(Dataset):
    def __init__(self, data_path, token, model_name_path, use_instruct):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_instruct = use_instruct

    def prompt_to_tokens(self, instruction, model_output):
        if self.use_instruct:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        else:
            tokens = tokenize_llama_base(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens

def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: List[str],
    model: LlamaWrapper,
):
    data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_instruct,
    )
    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)

        # debug_token_positions(dataset.tokenizer, p_tokens, "positive")
        
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            if p_activations.dim() == 3:
                p_activations = p_activations[0, -2, :].detach().cpu()
            elif p_activations.dim() == 2:
                p_activations = p_activations[-2, :].detach().cpu()
            else:
                raise ValueError(f"Unexpected activation tensor shape: {p_activations.shape}")
            pos_activations[layer].append(p_activations)
        
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            if n_activations.dim() == 3:
                n_activations = n_activations[0, -2, :].detach().cpu()
            elif n_activations.dim() == 2:
                n_activations = n_activations[-2, :].detach().cpu()
            else:
                raise ValueError(f"Unexpected activation tensor shape: {n_activations.shape}")
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            get_vector_path(behavior, layer, model.model_name_path),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )
        
        print(f"Saved vector for layer {layer}, shape: {vec.shape}")


def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    behaviors: List[str],
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model 
    behaviors: behaviors to generate vectors for
    """
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, use_instruct=not use_base_model
    )
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.behaviors
    )
