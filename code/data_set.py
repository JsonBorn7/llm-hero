import numpy as np
import os
import sentencepiece as spm
import sys
import torch


def train_model(fname, prefix):
    spm.SentencePieceTrainer.train(
        input=fname, model_prefix=prefix, vocab_size=16000,)
        # user_defined_symbols=['侠之大者', '为国为民'])

def load_file_into_splits(text_file, split_ratio):
    with open(text_file, 'r') as file:
        data = file.read()
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]

def load_tokenizer(model_file):
    sp = spm.SentencePieceProcessor()
    if not sp.load(model_file=model_file):
        return False, None
    else:
        return True, sp

def encode_and_save(sp, content, prefix):
    token_ids = sp.encode(content, out_type=int)
    print(f"data split of {prefix} has {len(token_ids)} tokens")
    token_ids = np.array(token_ids, dtype=np.int32)
    token_ids.tofile(os.path.join(os.path.dirname(__file__), "{}.dat".format(prefix)))

def gen_dataset(text_file, model_file):
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from: {model_file} failed")
        sys.exit(1)
    
    split_ratio = 0.9
    train_text, test_text = load_file_into_splits(text_file, split_ratio)
    encode_and_save(sp, train_text, "train")
    encode_and_save(sp, test_text, "test")

def get_batch(data):
    batch_size = 4
    block_size = 16
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int32)) for i in ix])
    return x, y

def test_samples():
    train_data = np.memmap(os.path.join("./", 'train.dat'), dtype=np.int32, mode='r')
    x, y = get_batch(train_data)

    model_file = "bird_shooter.model"
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from: {model_file} failed")
        sys.exit(1)
    
    # print(x, y)
    for features, targets in zip(x, y):
        print("feature:", sp.decode(features.tolist()))
        print("target:", sp.decode(targets.tolist()))


if __name__ == '__main__':
    train_model("bird_shooter.txt", "bird_shooter")
    gen_dataset("bird_shooter.txt", "bird_shooter.model")
    # test_samples()