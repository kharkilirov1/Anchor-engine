import torch

def calculate_metrics(logits, probs, old_id, new_id, tokenizer, top_k=10):
    """
    Given the model outputs, calculates rank, margin, and top-k inclusion.
    """
    p_old = probs[old_id].item()
    p_new = probs[new_id].item()
    
    logit_old = logits[old_id].item()
    logit_new = logits[new_id].item()
    margin = logit_new - logit_old
    
    sorted_indices = torch.argsort(logits, descending=True)
    
    rank_old = (sorted_indices == old_id).nonzero(as_tuple=True)[0].item() + 1
    rank_new = (sorted_indices == new_id).nonzero(as_tuple=True)[0].item() + 1
    
    in_top_k_old = rank_old <= top_k
    in_top_k_new = rank_new <= top_k
    
    # Get top 10 tokens for display
    top_10_indices = sorted_indices[:10]
    top_10_probs = probs[top_10_indices]
    
    top_10_table = []
    for i in range(10):
        token_str = tokenizer.decode(top_10_indices[i].item())
        prob_val = top_10_probs[i].item()
        top_10_table.append({"Rank": i+1, "Token": repr(token_str), "Probability": round(prob_val, 4)})
        
    return {
        "p_new": p_new,
        "p_old": p_old,
        "logit_new": logit_new,
        "logit_old": logit_old,
        "margin": margin,
        "rank_new": rank_new,
        "rank_old": rank_old,
        "in_top_k_new": in_top_k_new,
        "in_top_k_old": in_top_k_old,
        "top_10_table": top_10_table
    }
