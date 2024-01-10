import torch 
import torch.nn as nn
from torch.nn import functional as F
from Attentions.additive_attention import AdditiveAttention

# hyperparameters
batch_size = 32
block_size = 128
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 96    
n_head = 4
n_layer = 4
drop_out = 0.1
#---------------


torch.manual_seed(42)

with open('../Training texts/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All the characters in the text
chars = sorted(list(set(text)))
n_vocabs = len(chars)
# Map characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test split
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
test_data = data[split:]

# Load data
def get_batch(split):
    data = train_data if split == 'train' else test_data
    rand_idx = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[idx:idx+block_size] for idx in rand_idx])
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in rand_idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Single Head self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B * T * head_size)
        q = self.query(x) # (B * T * head_size)
        
        # wei = q @ k.transpose(-2, -1) * C**-.5 # (B * T * T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        attn = AdditiveAttention()
        
        v = self.value(x)
        out = attn(q,k,v)
        return out

class MultiHeadAttention(nn.Module):
    """Multi head self-attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeadForward(nn.Module):
    """Single Linear layer"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(drop_out))
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """A transformer block"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeadForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x
    

# The nanoGPT in build
class SimilarTextGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(n_vocabs, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocabs)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B * T * C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T * C)
        x = tok_emb + pos_emb # (B * T * C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B * T * n_vocabs)

        if targets == None:
            loss = None
        else:
            # Cross entropy needs C to be the second dimension
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond) 
            # Get only the last token
            logits = logits[:, -1, :] # (B * C)
            # Get probs using softmax
            probs = F.softmax(logits,dim=-1) # (B * C)
            # Generate the next index
            next_idx = torch.multinomial(probs, num_samples=1) # (B * 1)
            # appended the index to the original input so that it can be used to generate the next index
            idx = torch.cat((idx, next_idx), dim=1) # (B * T+1)
        
        return idx
    
model = SimilarTextGenerator().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for steps in range(max_iters):

    # print out loss during the training
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {steps}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

    # sample
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, 10000)[0].tolist()))
