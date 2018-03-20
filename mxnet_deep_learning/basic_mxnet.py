from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
model_ctx = mx.cpu()
# model_ctx = mx.gpu()

batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False,transform=transform), batch_size, shuffle=False)


class MLP(gluon.Block):
    def __init__(self, **kwargs):
        super(mlp,self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(63)
            self.dense1 = gluon.nn.Dense(63)
            self.dense2 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = nd.relu(self.dense2(x))
        x = self.dense2(x)
        return x



net = MLP()
net.collect_params().initialize(mx.init.Normal(sigma=01), ctx=model_ctx)

data = nd.ones((1,784))
net(data.as_in_context(model_ctx))




epochs = 10
smoothing_constant = .01



for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss+= nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = eval
