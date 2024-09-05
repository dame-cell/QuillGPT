# SmallGPT 
SmallGPT is a project focused on exploring and optimizing large language models (LLMs) by leveraging the GPT-2 architecture to generate text in the style of Shakespeare.

- We use the shakespere dataset usually used by Andrej karpathy because its easier to manage and its also quite small 
- The model is very small , with  134-Million  parameter 

# Examples 
The model does generate coherent text but it does not really grasp the full context of the prompt

| Prompt                                  | Generated Text                                                                 |
|-----------------------------------------|---------------------------------------------------------------------------------|
| First Citizen:Very well; and could be content to give him good report fort, but that he pays himself with being proud. Second Citizen:| Second Citizen:My lord, so is But your grace's a manhood,And in your grace that name,Your native king,And by their own have done:And with a Christian climate;And in their hearts. |
| All:Come, come.First Citizen:What, thou?  | Second Murderer:Why, then, then, then you a king?Second Murderer:Wherein your cousin?First Murderer:What is a duke?|
| All: |my lord, for our cousin,And that we'll prove so shall we are.KING RICHARD II:A man, as my heart.QUEEN MARGARET:So, my lord|


# Get Started 

You can try the model I train 
- First you might actually want to clone this repo 
```bash 
!git clone https://github.com/dame-cell/SmallGPT.git
cd SmallGPT
```
- Install the library 
```bash 
pip install -r requirements.txt 
cd model 
```
- Download the model from huggingface 

```python 
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="damerajee/smallgpt", filename="model.pt")
```

- And then try it out 
```python 
from generate import generate,text_to_token_ids , token_ids_to_text
from modeling_gpt2 import GPT2 ,GPTConfig 
import torch 

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_len": 256,
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
model = GPT2(GPT_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("path_to_model", map_location=device))
model.to(device)
import tiktoken 
tokenizer = tiktoken.get_encoding("gpt2")

text = "First citizen:"

token_ids = generate(
                    model=model,
                    device=device,
                    idx=text_to_token_ids(text, tokenizer),
                    max_new_tokens=50,
                    context_len=GPT_CONFIG["context_len"],
        )
        
print(token_ids_to_text(token_ids, tokenizer))
``` 
# Hyper-Parameters 

# Loss and Eval loss 

