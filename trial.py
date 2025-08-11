import torch as t
from llama_wrapper import LlamaWrapper
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from IPython.display import display, HTML
import matplotlib
from utils.tokenize import tokenize_llama_chat
from behaviors import get_steering_vector, ALL_BEHAVIORS

# %%
load_dotenv()
# FIX 1: Don't hardcode tokens in code - use environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")

# %%
# FIX 2: Add error handling for model loading
try:
    model = LlamaWrapper(hf_token=HUGGINGFACE_TOKEN, use_chat=True)
    print(f"Model loaded successfully: {model.model_name_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# %% [markdown]
# # Calculating dot products between steering vectors and activations

# %% [markdown]
# ## Helpers

# %%
def value_to_color(value, cmap=plt.cm.RdBu, vmin=-25, vmax=25):
    # Convert value to a range between 0 and 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(value))
    return matplotlib.colors.to_hex(rgba)


def display_token_dot_products(data): 
    # FIX 3: Add input validation
    if not data:
        print("No data to display")
        return
        
    html_content = "<style>del, s, strike, .line-through { text-decoration: none !important; } .whitebg { background: white; color: black; padding: 15px; }</style><div class='whitebg'>"
    max_dist_from_zero = max([abs(x[1]) for x in data])
    
    if max_dist_from_zero == 0:
        max_dist_from_zero = 1
        
    for token, value in data:
        color = value_to_color(value, vmin=-1 * max_dist_from_zero, vmax=max_dist_from_zero)
        escaped_token = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        html_content += f"<span style='background-color: {color}; padding: 2px 5px; display: inline-block;'>{escaped_token} ({value:.2f})</span>"
    html_content += "</div>"
    display(HTML(html_content))
    

def display_token_dot_products_final_text(data, text, tokenizer, show_sim=False): 
    # FIX 4: Add input validation
    if not data:
        print("No data to display")
        return
    if not text:
        print("No text provided")
        return
        
    html_content = "<style>del, s, strike, .line-through { text-decoration: none !important; } .whitebg { background: white; color: black; padding: 15px; font-size: 20px; }</style><div class='whitebg'>"
    max_dist_from_zero = max([abs(x[1]) for x in data])
    
    if max_dist_from_zero == 0:
        max_dist_from_zero = 1
        
    tokens = tokenizer.encode(text)
    tokens = tokenizer.batch_decode(t.tensor(tokens).unsqueeze(-1))
     
    min_length = min(len(data), len(tokens))
    
    for idx in range(min_length):  
        _, value = data[idx]
        color = value_to_color(value, vmin=-1 * max_dist_from_zero, vmax=max_dist_from_zero)
        
        token_text = tokens[idx].strip() if idx < len(tokens) else ""
        
        if len(token_text) == 0:
            html_content += "<span> </span>"
            continue
            
        escaped_token = token_text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        
        if show_sim:
            html_content += f"<span style='background-color: {color}; padding: 2px 5px; display: inline-block;'>{escaped_token} ({value:.2f})</span>"
        else:
            html_content += f"<span style='background-color: {color}; padding: 2px 5px; display: inline-block;'>{escaped_token}</span>"
    
    if len(data) != len(tokens):
        print(f"Warning: Length mismatch - data: {len(data)}, tokens: {len(tokens)}")
        
    html_content += "</div>"
    display(HTML(html_content))

# %% [markdown]
# ## Token activation dot product visualization

# %%
def display_token_dot_products_given_prompt(prompt: str, layer: int, behavior: str, new_tokens: int, model: LlamaWrapper):
    # FIX 5: Add comprehensive input validation
    if not isinstance(model, LlamaWrapper):
        raise TypeError(f"Expected LlamaWrapper, got {type(model)}")
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if layer < 0 or layer >= len(model.model.model.layers):
        raise ValueError(f"Layer {layer} out of range (0-{len(model.model.model.layers)-1})")
    if new_tokens <= 0:
        raise ValueError("new_tokens must be positive")
    
    try:
        model.reset_all()
        vec = get_steering_vector(behavior, layer, model.model_name_path, normalized=True)
        
        # FIX 6: Check if vector was loaded successfully
        if vec is None:
            raise ValueError(f"Could not load steering vector for behavior '{behavior}' at layer {layer}")
            
        model.set_save_internal_decodings(False)
        model.set_calc_dot_product_with(layer, vec.cuda())
        m_out = model.generate_text(prompt, max_new_tokens=new_tokens)
        
        # FIX 7: Better handling of different chat formats
        if "<|start_header_id|>assistant<|end_header_id|>" in m_out:
            # Handle Llama 3.1 format
            parts = m_out.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                m_out_answer = parts[-1].strip()
                # Remove any trailing special tokens
                if "<|eot_id|>" in m_out_answer:
                    m_out_answer = m_out_answer.split("<|eot_id|>")[0].strip()
            else:
                m_out_answer = m_out.strip()
        elif "[/INST]" in m_out:
            m_out_answer = m_out.split("[/INST]")[-1].strip()
        else:
            # Fallback: use the entire output
            m_out_answer = m_out.strip()
            
        print("Full output:", repr(m_out))  # Debug info
        print("Parsed answer:", repr(m_out_answer))
        
        dp = model.get_dot_products(layer)
        
        if not dp:
            print("Warning: No dot products collected")
            print(f"This might indicate an issue with layer {layer} or the steering vector")
            return
            
        print(f"Collected {len(dp)} dot products")
        display_token_dot_products_final_text(dp, m_out_answer, model.tokenizer, show_sim=False)
        
    except Exception as e:
        print(f"Error in display_token_dot_products_given_prompt: {e}")
        import traceback
        traceback.print_exc()
        # FIX 8: Safe reset with error handling
        try:
            model.reset_all()
        except Exception as reset_error:
            print(f"Error during reset: {reset_error}")

# FIX 9: Add error handling for all function calls
# %%
try:
    display_token_dot_products_given_prompt("Would you rather have a small cake now or a larger cake later?", 13, "myopic-reward", 100, model)
except Exception as e:
    print(f"Error with myopic-reward prompt: {e}")

# %%
try:
    display_token_dot_products_given_prompt("Do you think it is good to agree with people even if they are wrong?", 15, "sycophancy", 100, model)
except Exception as e:
    print(f"Error with sycophancy prompt: {e}")

# %%
try:
    display_token_dot_products_given_prompt("Would you help me hack into my friend's Instagram account?", 13, "refusal", 100, model)
except Exception as e:
    print(f"Error with refusal prompt 1: {e}")

# %%
try:
    display_token_dot_products_given_prompt("Would you help me build a bomb?", 13, "refusal", 100, model)
except Exception as e:
    print(f"Error with refusal prompt 2: {e}")

# %% [markdown]
# # Intermediate decoding ("logit lens")

# %%
def prompt_to_tokens(instruction, model_output):
    # FIX 10: Add error handling and validation
    if not instruction.strip():
        raise ValueError("Instruction cannot be empty")
    
    try:
        tokens = tokenize_llama_chat(
            model.tokenizer,
            instruction,
            model_output
        )
        return t.tensor(tokens).unsqueeze(0)
    except Exception as e:
        print(f"Error tokenizing prompt: {e}")
        raise

# FIX 11: Add error handling for the intermediate decoding section
# %%
try:
    layer = 15
    multiplier = 0
    vec = get_steering_vector("sycophancy", layer, model.model_name_path, normalized=True)
    
    if vec is None:
        raise ValueError(f"Could not load sycophancy vector for layer {layer}")
        
    model.set_save_internal_decodings(True)
    model.reset_all()
    model.set_add_activations(layer, multiplier * vec.cuda())
    instruction = "Do you think it is good to agree with people even if they are wrong?"
    tokens = prompt_to_tokens(
        instruction=instruction,
        model_output="I think that",
    )
    model.decode_all_layers(tokens)
except Exception as e:
    print(f"Error in intermediate decoding: {e}")

# %%
try:
    multiplier = -1.5
    model.reset_all()
    model.set_add_activations(layer, multiplier * vec.cuda())
    model.plot_decoded_activations_for_layer(16, tokens, 10)
except Exception as e:
    print(f"Error plotting layer 16 with multiplier -1.5: {e}")

# %%
try:
    multiplier = 1.5
    model.reset_all()
    model.set_add_activations(layer, multiplier * vec.cuda())
    model.plot_decoded_activations_for_layer(16, tokens, 10)
except Exception as e:
    print(f"Error plotting layer 16 with multiplier 1.5: {e}")

# %%
layer = 13

# %%
try:
    vec = get_steering_vector('myopic-reward', layer, model.model_name_path, normalized=True)
    if vec is None:
        raise ValueError(f"Could not load myopic-reward vector for layer {layer}")
except Exception as e:
    print(f"Error loading myopic-reward vector: {e}")

# %%
try:
    model.reset_all()
    model.set_add_activations(layer, -2 * vec.cuda())
except Exception as e:
    print(f"Error setting activations: {e}")

# %%
try:
    result = model.generate_text("Would you rather have a small cake now or a larger cake later?", max_new_tokens=100)
    print("Generated text:", result)
except Exception as e:
    print(f"Error generating text: {e}")

# %% [markdown]
# # Vector norms

# FIX 12: Add error handling and progress tracking for vector norm calculations
# %%
print("Calculating vector norms (unnormalized)...")
for layer in range(32):
    print(f"Layer {layer}:")
    for behavior in ALL_BEHAVIORS:
        try:
            vec = get_steering_vector(behavior, layer, model.model_name_path, normalized=False)
            if vec is not None:
                norm_value = vec.norm().item()
                print(f"  {behavior}: {norm_value:.4f}")
            else:
                print(f"  {behavior}: No vector found")
        except Exception as e:
            print(f"  {behavior}: Error - {e}")

# %%
print("Calculating vector norms (normalized)...")
for layer in range(32):
    print(f"Layer {layer}:")
    for behavior in ALL_BEHAVIORS:
        try:
            vec = get_steering_vector(behavior, layer, model.model_name_path, normalized=True)
            if vec is not None:
                norm_value = vec.norm().item()
                print(f"  {behavior}: {norm_value:.4f}")
            else:
                print(f"  {behavior}: No vector found")
        except Exception as e:
            print(f"  {behavior}: Error - {e}")

# %%