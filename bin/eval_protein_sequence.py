import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from alphabets import Uniprot21

def parse_2line(f):
    names = []
    xs = []
    ys = []
    for line in f:
        if line.startswith(b'>'):
            name = line[1:]
            # get the sequence
            x = f.readline().strip()
            names.append(name)
            xs.append(x)
    return names, xs

def load_2line(path, alphabet):
    with open(path, 'rb') as f:
        names, x = parse_2line(f)
    x = [alphabet.encode(x) for x in x]
    return names, x

def load_data():
    alphabet = Uniprot21()

    path = 'data/davis2011kinase/prot_sequences.fasta'
    names, x = load_2line(path, alphabet)

    datasets = {'dataset0': (x, names)}

    return datasets

def split_dataset(xs, ys, random=np.random, k=5):
    x_splits = [[] for _ in range(k)]
    y_splits = [[] for _ in range(k)]
    order = random.permutation(len(xs))
    for i in range(len(order)):
        j = order[i]
        x_s = x_splits[i%k]
        y_s = y_splits[i%k]
        x_s.append(xs[j])
        y_s.append(ys[j])
    return x_splits, y_splits

def unstack_lstm(lstm):
    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
            #setattr(layer, dest, getattr(lstm, src))

            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
            #setattr(layer, dest, getattr(lstm, src))
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers

def featurize(x, lm_embed, lstm_stack, proj, include_lm=True, lm_only=False):
    zs = []

    x_onehot = x.new(x.size(0),x.size(1), 21).float().zero_()
    x_onehot.scatter_(2,x.unsqueeze(2),1)
    zs.append(x_onehot)

    h = lm_embed(x)
    if include_lm:
        zs.append(h)
    if not lm_only:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            zs.append(h)
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)
    z = torch.cat(zs, 2)
    return z

def featurize_dict(datasets, lm_embed, lstm_stack, proj,
                   use_cuda=False, include_lm=True, lm_only=False):
    z = {}
    for k,v in datasets.items():
        x_k = v[0]
        z[k] = []
        with torch.no_grad():
            for x in x_k:
                x = torch.from_numpy(x).long().unsqueeze(0)
                if use_cuda:
                    x = x.cuda()
                z_x = featurize(x, lm_embed, lstm_stack, proj,
                                include_lm=include_lm, lm_only=lm_only)
                z_x = z_x.squeeze(0).cpu()
                z[k].append(z_x)
    return z


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to saved embedding model')
    parser.add_argument('--hidden-dim', type=int, default=150, help='dimension of LSTM (default: 150)')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs (default: 10)')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    args = parser.parse_args()

    datasets = load_data()
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim

    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)


    ## load the embedding model
    encoder = torch.load(args.model)
    encoder.eval()
    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    if use_cuda:
        lm_embed.cuda()
        for lstm in lstm_stack:
            lstm.cuda()
        proj.cuda()

    ## featurize the sequences
    z = featurize_dict(datasets, lm_embed, lstm_stack, proj, use_cuda=use_cuda)

    embeddings = z['dataset0']
    names = datasets['dataset0'][1]
    assert(len(embeddings) == len(names))
    for name, embedding in zip(names, embeddings):
        print('>{}'.format(name))
        print('\t'.join([ str(val) for val in embedding.mean(0) ]))

if __name__ == '__main__':
    main()
