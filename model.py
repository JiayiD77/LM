import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#---------------

torch.manual_seed(42)

with open('input.txt', 'r', encoding='utf-8') as f:
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

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B * T * head_size)
        q = self.query(x) # (B * T * head_size)
        
        wei = q @ k.transpose(-2, -1) * C**-.5 # (B * T * T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei@v
        return out

# The baseline bigram model
class Improving(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(n_vocabs, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocabs)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B * T * C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T * C)
        x = tok_emb + pos_emb # (B * T * C)
        x = self.sa_head(x)
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
    
model = Improving().to(device)

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
print(decode(model.generate(context, 1000)[0].tolist()))
