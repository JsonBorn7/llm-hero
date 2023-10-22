from gpt_model import GPTConfig, GPTModel

import numpy as np
import sentencepiece as spm
import sys
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate = 1e-3
max_iters = 12000

train_data = np.memmap('train.dat', dtype=np.int32, mode='r')
test_data = np.memmap('test.dat', dtype=np.int32, mode='r')

def get_batch(split, config):
    data = train_data if split == 'train' else test_data 
    ix = torch.randint(len(data) - config.seq_len, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.seq_len]).astype(np.int32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.seq_len]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def train(config, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter_num in range(max_iters):
        optimizer.zero_grad()

        xb, yb = get_batch('train', config)

        # forward and loss caculation
        _, loss = model(xb, yb)
        if (iter_num + 1) % 100  == 0:
            print(f"[train_info] iter:{iter_num+1:5d}, loss:{loss.item():5.3f}")
        
        # backward and gradient descent
        loss.backward()

        # update weights
        optimizer.step()

    print(f"final loss: {loss.item()}")

        
def main():
    config = GPTConfig()
    config.batch_size = 32
    config.dropout = 0.1

    model = GPTModel(config).to(device)

    train(config, model)

    # load tokenizer
    from data_set import load_tokenizer
    model_file = "bird_shooter.model"
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from: {model_file} failed")
        sys.exit(1)

    # generate from the model
    user_inputs = ["郭靖一掌挥出", "黄蓉突然想到", "周伯通好奇心大起", "洪七公哈哈大笑"]
    for user_input in user_inputs:
        context = torch.tensor([sp.encode(user_input)], dtype=torch.int32, device=device)
        gpt_output = model.generate(context, max_new_tokens=50)[0].tolist()
        print(f"gpt({user_input}) => {sp.decode(gpt_output)}")

if __name__ == '__main__':
    main()