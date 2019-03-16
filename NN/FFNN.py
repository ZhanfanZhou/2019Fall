import numpy as np


class FeedForwardNN:

    def __init__(self, feature_dim, xs, ys, seed=6):
        input_dim = len(xs[0])
        output_dim = len(ys[0])
        self.xs = xs
        self.ys = ys
        self.ws = self.init_weights(input_dim, output_dim, feature_dim, seed)

    def init_weights(self, input_dim, output_dim, feature_dim, seed):
        np.random.seed(seed)
        layers = [input_dim] + feature_dim + [output_dim]
        return [np.random.rand(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(x))

    def softmax(self, x):
        e = np.exp(x)
        return e / e.sum()

    def feed_forward(self, x, w):
        net = np.dot(x, w)
        return self.sigmoid(net)

    def back_propagation(self, delta_error, out, x, weight, learning=0.9):
        # print "x", x.shape
        # print "weight", weight.shape
        delta = delta_error * self.sigmoid(out, derivative=True)
        # print "delta", delta.shape
        weight = weight - learning * np.dot(x.T, delta)
        return weight, np.dot(delta, weight.T)

    def train_one(self, x, y):
        ys = [y]
        outs = []
        for i in range(len(self.ws)):
            outs.append(x)
            x = self.feed_forward(x, self.ws[i])
        outs.append(self.softmax(x))
        print "Layers: ", outs
        loss = 0.5 * np.square(ys[-1] - outs[-1]).sum()
        print "Loss: %s" % loss
        ys[-1] = outs[-1] - ys[-1]
        for i in range(1, len(self.ws)+1):
            self.ws[-i], new_y = self.back_propagation(ys[-1], outs[-i], outs[-i-1], self.ws[-i])
            ys.append(new_y)
        # print self.ws

    def train(self):
        for x, y in zip(self.xs, self.ys):
            nn.train_one(x.reshape(1, x.shape[0]), y)

    def test(self, xs, ys):
        for x, y in zip(xs, ys):
            for i in range(len(self.ws)):
                x = self.feed_forward(x, self.ws[i])
            softmax = self.softmax(x)
            loss = 0.5 * np.square(y - softmax).sum()


if __name__ == "__main__":
    inputs = [np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0]),
              np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
              np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1])]
    targets = [np.array([0, 1, 0, 0]),
               np.array([0, 0, 1, 0]),
               np.array([1, 0, 0, 0])]
    FeedForwardNN([7, 7, 7], inputs, targets).train()
