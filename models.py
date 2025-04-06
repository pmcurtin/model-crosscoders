from torch import nn

class CrossCoder(nn.Module):

    def __init__(self):
        pass

    def encode(self, x, m):
        # has two encoders, so m indicates which to use
        pass

    def decode(self, x, m):
        pass

    def forward(self, x):
        # encode, then decode
        # do we need this?
        pass

    def loss(self, x, y):
        # compute loss.
        # losses from: x -> x, x -> y, y -> x, y -> y reconstructions
        pass

    # save/load too

# I like Neel Nanda's buffer method for generating residuals on-the-fly
# read his impl first. Ours will be simpler probably
class ResidualBuffer():

    def __init__(self):
        # here, we will have two models, not one.
        pass

    def next(self):
        # return a batch of residual streams
        pass

    def refresh(self):
        # initally fill the buffer (first call)
        # then just discard old examples and top it up on subsequent calls
        # we can use transformerlens here (easy, use activation cache), or some fancier way of getting 
        # the residuals that we want (for example, no need to execute transformer past 
        # layer of interest)
        pass


class ModelWithEncoder(nn.Module):
    # I think delphi needs a model that takes in text and spits out
    # SAE features. This can be that model

    def __init__(self, model, encoder, layer):
        pass

    def forward(self, x):
        # first run model on x
        # then run encoder on residual
        pass


