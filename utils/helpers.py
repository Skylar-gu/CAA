import torch as t
import matplotlib.pyplot as plt

def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)


def add_vector_from_position(matrix, vector, position_ids, from_pos=None):
    """
    Add a vector to matrix starting from a specific position
    """
    # Create a copy to avoid in-place modification
    result_matrix = matrix.clone()
    
    from_id = from_pos
    if from_id is None:
        from_id = position_ids.min().item() - 1

    # Handle position_ids dimensions - extract the actual position sequence
    if position_ids.dim() == 2:
        # Take first batch: [1, 221] -> [221]
        position_sequence = position_ids[0]
    elif position_ids.dim() == 1:
        # Already 1D: [221]
        position_sequence = position_ids
    else:
        raise ValueError(f"Unsupported position_ids dimension: {position_ids.dim()}")

    # Handle matrix dimensions
    if matrix.dim() == 2:
        # Shape: [seq_len, hidden_dim] = [221, 4096]
        seq_len, hidden_dim = matrix.shape
        
        # Create mask: [seq_len] = [221]
        mask = position_sequence >= from_id
        mask = mask.unsqueeze(-1)  # [221, 1]
        
        # Ensure vector is on same device: [4096]
        vector = vector.to(matrix.device)
        
        # Broadcasting: [221, 1] * [4096] -> [221, 4096]
        result_matrix += mask.float() * vector
        
    elif matrix.dim() == 3:
        # Shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = matrix.shape
        
        # Expand position_sequence to match batch if needed
        if position_sequence.size(0) != seq_len:
            raise ValueError(f"position_ids sequence length {position_sequence.size(0)} doesn't match matrix seq_len {seq_len}")
        
        # Create mask and expand for batch dimension
        mask = position_sequence >= from_id  # [seq_len]
        mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        mask = mask.expand(batch_size, -1, -1)  # [batch_size, seq_len, 1]
        
        vector = vector.to(matrix.device)  # [hidden_dim]
        
        # Broadcasting: [batch_size, seq_len, 1] * [hidden_dim] -> [batch_size, seq_len, hidden_dim]
        result_matrix += mask.float() * vector
        
    else:
        raise ValueError(f"Unsupported matrix dimension: {matrix.dim()}. Expected 2D or 3D.")
    
    return result_matrix


def find_last_subtensor_position(tensor, sub_tensor):
    """
    Find the last occurrence of sub_tensor in tensor
    """
    # Ensure tensors are on the same device and have same dtype
    if tensor.device != sub_tensor.device:
        sub_tensor = sub_tensor.to(tensor.device)
    
    if tensor.dtype != sub_tensor.dtype:
        sub_tensor = sub_tensor.to(tensor.dtype)
    
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n or m == 0:
        return -1
    
    # Search from the end backwards
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    """
    Find where the instruction ends and assistant response begins
    """
    if tokens.dim() > 1:
        # Handle batch dimension
        tokens = tokens[0] if tokens.size(0) == 1 else tokens
    
    if end_str.dim() > 1:
        end_str = end_str[0] if end_str.size(0) == 1 else end_str
    
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        # If end_str not found, return a reasonable default
        print(f"Warning: end_str not found in tokens. Returning last position.")
        return len(tokens) - 1
    
    end_pos = start_pos + len(end_str) - 1
    
    # Ensure we don't go out of bounds
    return min(end_pos, len(tokens) - 1)


def get_a_b_probs(logits, a_token_id, b_token_id):
    """
    Get probabilities for tokens A and B from logits
    """
    # Input validation
    if logits.dim() != 3:
        raise ValueError(f"Expected logits to be 3D [batch, seq, vocab], got shape {logits.shape}")
    
    if logits.size(0) != 1:
        raise ValueError(f"Expected batch size 1, got {logits.size(0)}")
    
    vocab_size = logits.size(-1)
    
    # Validate token IDs
    if not (0 <= a_token_id < vocab_size):
        raise ValueError(f"a_token_id {a_token_id} out of vocabulary range [0, {vocab_size})")
    
    if not (0 <= b_token_id < vocab_size):
        raise ValueError(f"b_token_id {b_token_id} out of vocabulary range [0, {vocab_size})")
    
    # Get last token logits
    last_token_logits = logits[0, -1, :]
    
    # Apply softmax with numerical stability
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    
    # Extract probabilities
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    
    return a_prob, b_prob

def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'


def get_model_path(is_base: bool):
    if is_base:
        return "meta-llama/Meta-Llama-3.1-8B"
    else:
        return "meta-llama/Meta-Llama-3.1-8B-Instruct"

def model_name_format(name: str) -> str:
    name = name.lower()
    is_instruct = "instruct" in name   
    if is_instruct:
        return "Llama 3.1 8B Instruct"
    else:
        return "Llama 3.1 8B"


def debug_tensor_shapes(matrix, vector, position_ids, from_pos=None):
    """Debug helper to understand tensor shapes"""
    print("=== Tensor Shape Debug ===")
    print(f"matrix.shape: {matrix.shape}")
    print(f"vector.shape: {vector.shape}")
    print(f"position_ids.shape: {position_ids.shape}")
    print(f"from_pos: {from_pos}")
    print(f"matrix.device: {matrix.device}")
    print(f"vector.device: {vector.device}")
    print(f"position_ids.device: {position_ids.device}")
    print("="*30)