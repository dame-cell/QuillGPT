import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from tqdm.notebook import tqdm 
import time 
from modeling_gpt2 import GPT2 
from utils import create_dataloader_v1 

GPT_CONFIG= {
    "vocab_size": 50257,    
    "context_length": 512, 
    "emb_dim": 384,        
    "n_heads": 4,          
    "n_layers": 6,         
    "drop_rate": 0.1,      
    "qkv_bias": False,
    "epochs":4,
    "learning_rate":0.01,
           } 
