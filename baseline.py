import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

# The baseline bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx) # (B*T*C)

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
            logits, loss = self(idx) 
            # Get only the last token
            logits = logits[:, -1, :] # (B * C)
            # Get probs using softmax
            probs = F.softmax(logits,dim=-1) # (B * C)
            # Generate the next index
            next_idx = torch.multinomial(probs, num_samples=1) # (B * 1)
            # appended the index to the original input so that it can be used to generate the next index
            idx = torch.cat((idx, next_idx), dim=1) # (B * T+1)
        
        return idx
    
model = BigramLanguageModel(vocab_size=n_vocabs).to(device)

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
