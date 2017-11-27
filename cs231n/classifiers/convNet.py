import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvNet(object):
    '''
    [conv - relu - conv - relu - pool(2*2)] * 2 - [affine] * 2 - softmax
    '''

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros((1, num_filters))

        self.params['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b2'] = np.zeros((1, num_filters))

        self.params['W3'] = weight_scale * np.random.randn(num_filters//2, num_filters, filter_size, filter_size)
        self.params['b3'] = np.zeros((1, num_filters//2))

        self.params['W4'] = weight_scale * np.random.randn(num_filters//2, num_filters//2, filter_size, filter_size)
        self.params['b4'] = np.zeros((1, num_filters//2))

        self.params['W5'] = weight_scale * np.random.randn((num_filters//2)*(H//4)*(W//4), hidden_dim)
        self.params['b5'] = np.zeros((1, hidden_dim))

        self.params['W6'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b6'] = np.zeros((1, num_classes))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']

        filter1_size = W1.shape[2]
        filter2_size = W2.shape[2]
        filter3_size = W3.shape[2]
        filter4_size = W4.shape[2]

        conv1_param = {'stride': 1, 'pad': (filter1_size - 1) // 2}
        conv2_param = {'stride': 1, 'pad': (filter2_size - 1) // 2}
        conv3_param = {'stride': 1, 'pad': (filter3_size - 1) // 2}
        conv4_param = {'stride': 1, 'pad': (filter4_size - 1) // 2}

        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        a1, cache1 = conv_relu_forward(X, W1, b1, conv1_param)
        a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv2_param, pool_param)
        a3, cache3 = conv_relu_forward(a2, W3, b3, conv3_param)
        a4, cache4 = conv_relu_pool_forward(a3, W4, b4, conv4_param, pool_param)
        a5, cache5 = affine_forward(a4, W5, b5)
        scores, cache6 = affine_forward(a5, W6, b6)

        if y is None:
            return scores

        data_loss, dscores = softmax_loss(scores, y)
        da5, dW6, db6 = affine_backward(dscores, cache6)
        da4, dW5, db5 = affine_backward(da5, cache5)
        da3, dW4, db4 = conv_relu_pool_backward(da4, cache4)
        da2, dW3, db3 = conv_relu_backward(da3, cache3)
        da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
        dX, dW1, db1 = conv_relu_backward(da1, cache1)

        #Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        dW6 += self.reg * W6

        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1,W2,W3,W4,W5,W6])
        loss = data_loss + reg_loss

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3,
                 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6}

        return loss, grads
