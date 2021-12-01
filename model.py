#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: model.py
# =====================================
import pprint

import torch
import numpy as np
from torch import nn
from torch.nn import Sequential
from torch.autograd import Variable

# tf.config.experimental.set_visible_devices([], 'GPU')
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)


def get_activation_func(key: str):
    assert isinstance(key, str)

    activation_func = None
    if key == 'gelu':
        activation_func = nn.GELU

    elif key == 'relu':
        activation_func = nn.ReLU

    elif key == 'tanh':
        activation_func = nn.Tanh

    elif key == 'softplus':
        activation_func = nn.Softplus

    elif key == 'linear':
        activation_func = nn.Identity

    if activation_func is None:
        print('input activation name:' + key)
        raise RuntimeError

    return activation_func


class LinearReLU(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearReLU, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=1,
            padding=0,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        hidden = self.conv(x)
        out = self.relu(hidden)
        return out

    # # Operator fusion
    # def fuse_modules(self):
    #     torch.quantization.fuse_modules(
    #         self,
    #         ["conv", "relu"],
    #         inplace=True,
    #         fuser_func=quantization.fuse_known_modules,
    #     )


def mlp_hidden(num_hidden_layers, num_hidden_units):
    layers = []
    for j in range(num_hidden_layers):
        layers += [LinearReLU(num_hidden_units, num_hidden_units)]
    return nn.Sequential(*layers)


class MLPNet(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet, self).__init__()  # name=kwargs['name']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = kwargs['name'] if kwargs.get('name') else 'unknown_network'
        self.cat = torch.nn.quantized.FloatFunctional()

        self.first_layer = LinearReLU(input_dim, num_hidden_units)
        self.hidden_layers = mlp_hidden(num_hidden_layers, num_hidden_units)
        self.last_layer = torch.nn.Conv2d(in_channels=num_hidden_units, out_channels=output_dim, kernel_size=1, padding=0)

        out_act = get_activation_func(kwargs['output_activation']) if kwargs.get('output_activation') else nn.Identity
        self.last_activation = out_act()

        self.init_weights()

    def forward(self, input):
        # print('11', input)
        list_input = torch.chunk(input, self.input_dim, dim=1)
        quantized_list_input = []
        for inp in list_input:
            inp = inp.unsqueeze(-1).unsqueeze(-1)
            # quantized_inp = self.quant(inp)
            quantized_list_input.append(inp)
        quantized_input = self.cat.cat(quantized_list_input, dim=1)
        # print('22', quantized_input.shape)
        output_first_hidden = self.first_layer(quantized_input)
        output_last_hidden = self.hidden_layers(output_first_hidden)
        output_quantized = self.last_layer(output_last_hidden)

        # output = self.dequant(output_quantized)  # 量化训练直接输出
        output = output_quantized.view(-1, self.output_dim)
        output = self.last_activation(output)
        return output

    def init_weights(self):
        for layer in self.first_layer.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2.))
        for linear_relu in self.hidden_layers:
            for layer in linear_relu.children():
                if isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2.))
        if isinstance(self.last_layer, nn.Conv2d):
            nn.init.orthogonal_(self.last_layer.weight.data, gain=1.)
            nn.init.constant_(self.last_layer.bias.data, 0)


class MLPNet1(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet1, self).__init__()  # name=kwargs['name']
        self.name = kwargs['name'] if kwargs.get('name') else 'unknown_network'

        act = get_activation_func(hidden_activation)
        out_act = get_activation_func(kwargs['output_activation']) if kwargs.get('output_activation') else nn.Identity
        hidden_layers = []
        for _ in range(num_hidden_layers - 1):
            hidden_layers += [torch.nn.Linear(num_hidden_units, num_hidden_units), act()]

        self.first_ = Sequential(torch.nn.Linear(input_dim, num_hidden_units), act())
        self.hidden = Sequential(*hidden_layers)
        self.output = Sequential(torch.nn.Linear(num_hidden_units, output_dim), out_act())

        self.init_weights()

    def forward(self, x):
        x = self.first_(x)
        x = self.hidden(x)
        x = self.output(x)
        return x

    def init_weights(self):
        for layer in self.first_.children():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2.))
        for layer in self.hidden.children():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2.))
        for layer in self.output.children():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight.data, gain=1.)
                nn.init.constant_(layer.bias.data, 0)


class MLPNet2(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet2, self).__init__()  # name=kwargs['name']
        act = get_activation_func(hidden_activation)
        hidden_layers = [torch.nn.Linear(input_dim, num_hidden_units), act()]
        for _ in range(num_hidden_layers - 1):
            hidden_layers += [torch.nn.Linear(num_hidden_units, num_hidden_units), act()]
        out_act = get_activation_func(kwargs['output_activation']) if kwargs.get('output_activation') else nn.Identity
        layers = hidden_layers + [torch.nn.Linear(num_hidden_units, output_dim), out_act()]
        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def test_attrib():
    a = Variable(0)

    p = MLPNet(2, 2, 128, 1, name='ttt')
    print(hasattr(p, 'get_weights'))
    print(hasattr(p, 'trainable_weights'))
    print(hasattr(a, 'get_weights'))
    print(hasattr(a, 'trainable_weights'))
    print(type(a))
    print(type(p))
    # print(a.name)
    # print(p.name)
    # p.build((None, 2))
    p.summary()
    # inp = np.random.random([10, 2])
    # out = p.forward(inp)
    # print(p.get_weights())
    # print(p.trainable_weights)


def test_out():
    actor = MLPNet(2, 3, 2, 'relu', 1, name='actor', output_activation='linear')
    pprint.pprint(actor)
    for name, param in actor.named_parameters():
        print(f'{name} = {param}')
    state = torch.tensor([0.4858, 0.9345]).float()
    out = actor.forward(state)
    print(f'state = {state}, out = {out}')


def test_copy():
    actor1 = MLPNet(2, 3, 2, 'relu', 1, name='actor', output_activation='linear')
    actor2 = MLPNet(2, 3, 2, 'relu', 1, name='actor', output_activation='linear')
    print('old actor1: ')
    for name, param in actor1.named_parameters():
        print(f'{name} = {param}')
    print('old actor2: ')
    for name, param in actor2.named_parameters():
        print(f'{name} = {param}')

    weight = actor1.state_dict()
    actor2.load_state_dict(weight)

    print('new actor1: ')
    for name, param in actor1.named_parameters():
        print(f'{name} = {param}')
    print('new actor2: ')
    for name, param in actor2.named_parameters():
        print(f'{name} = {param}')


def test_split():
    a = torch.rand(2, 4)
    print(a)
    a1, a2 = torch.chunk(a, chunks=2, dim=-1)
    print(a1)
    print(a2)


def test_mlp():
    from train_script import built_parser
    args = built_parser('AMPC')

    obs_dim, act_dim = args.obs_dim, args.act_dim
    n_hiddens, n_units, hidden_activation = args.num_hidden_layers, args.num_hidden_units, args.hidden_activation
    policy = MLPNet(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                    output_activation=args.policy_out_activation)
    print(policy(torch.rand(2, 80)))
    policy1 = MLPNet1(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                      output_activation=args.policy_out_activation)
    print(policy1(torch.rand(2, 80)))


if __name__ == '__main__':
    test_mlp()
