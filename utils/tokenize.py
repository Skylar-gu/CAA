from transformers import PreTrainedTokenizer
from typing import List, Optional

# Llama 3.1 special tokens
BEGIN_TEXT = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"
END_TEXT = "<|end_of_text|>"

# Base model tokens (if you still need base model support)
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

# Position markers for intervention
ADD_FROM_POS_CHAT = f"{START_HEADER}assistant{END_HEADER}\n\n"
ADD_FROM_POS_BASE = BASE_RESPONSE


def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    """
    Tokenize for Llama 3.1 chat format
    Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {model_output}
    """
    messages = []
    
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_input.strip()})
    
    if model_output is not None:
        messages.append({"role": "assistant", "content": model_output.strip()})

    return tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True if model_output is None else False
    )


def tokenize_llama_base(
    tokenizer: PreTrainedTokenizer, 
    user_input: str, 
    model_output: str = None
) -> List[int]:
    """
    Tokenize for base model format (unchanged from Llama 2)
    """
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_llama3_manual(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    """
    Manual tokenization for Llama 3.1 if you prefer not to use apply_chat_template
    """
    input_content = BEGIN_TEXT
    
    if system_prompt is not None:
        input_content += f"{START_HEADER}system{END_HEADER}\n{system_prompt.strip()}{EOT}"
    
    input_content += f"{START_HEADER}user{END_HEADER}\n{user_input.strip()}{EOT}"
    input_content += f"{START_HEADER}assistant{END_HEADER}\n"
    
    if model_output is not None:
        input_content += f"{model_output.strip()}"
    
    return tokenizer.encode(input_content)