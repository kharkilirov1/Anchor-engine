import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

TOP_SELECTIVE_HEADS = [
    (0, 4),   # Layer 0, Head 4
    (1, 4),   # Layer 1, Head 4
    (19, 12), # Layer 19, Head 12
    (23, 4),  # Layer 23, Head 4
    (5, 11)   # Layer 5, Head 11
]

CONTROL_HEADS = [
    (2, 1),
    (3, 6),
    (8, 3),
    (11, 13)
]

class InterventionContext:
    """Hooks into the attention output to scale specific heads."""
    def __init__(self, target_heads, scale_factor=1.0):
        self.target_heads = target_heads
        self.scale_factor = scale_factor
        self.handles = []
        
    def attach(self, model):
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.num_heads
        
        layer_to_heads = defaultdict(list)
        for l, h in self.target_heads:
            layer_to_heads[l].append(h)
            
        def create_pre_o_proj_hook(heads_to_intervene):
            def hook(module, args):
                hidden_states = args[0].clone()
                B, T, HD = hidden_states.shape
                hidden_states = hidden_states.view(B, T, self.num_heads, self.head_dim)
                
                for h_idx in heads_to_intervene:
                    hidden_states[:, :, h_idx, :] *= self.scale_factor
                        
                hidden_states = hidden_states.view(B, T, HD)
                return (hidden_states,)
            return hook

        for l_idx, heads in layer_to_heads.items():
            handle = model.model.layers[l_idx].self_attn.o_proj.register_forward_pre_hook(create_pre_o_proj_hook(heads))
            self.handles.append(handle)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

class ModelController:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def load_model(self):
        if self.model is None:
            print("Loading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            print("Loading Model (CPU)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                device_map="cpu", 
                torch_dtype=torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            self.model.eval()
            print("Model loaded successfully.")

    def predict_next_token(self, prompt, old_str, new_str, mode="Baseline", scale=1.0):
        """
        Mode can be: 'Baseline', 'Selective', 'Control'
        """
        self.load_model()
        
        target_heads = []
        if mode == "Selective":
            target_heads = TOP_SELECTIVE_HEADS
        elif mode == "Control":
            target_heads = CONTROL_HEADS
            
        ctx = None
        if target_heads and scale != 1.0:
            ctx = InterventionContext(target_heads, scale)
            ctx.attach(self.model)
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Generate a few more tokens for context
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            
        if ctx:
            ctx.detach()
            
        full_text = self.tokenizer.decode(gen_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
        next_token_logits = outputs.logits[0, -1, :] # [Vocab_size]
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Determine target token IDs (taking the first token of the string)
        old_ids = self.tokenizer.encode(old_str, add_special_tokens=False)
        new_ids = self.tokenizer.encode(new_str, add_special_tokens=False)
        
        old_id = old_ids[0] if old_ids else 0
        new_id = new_ids[0] if new_ids else 0
        
        return next_token_logits, probs, old_id, new_id, old_str, new_str, full_text
