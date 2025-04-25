from neural import *

xor_data = [([1, 1], [0]),  ([1, 0], [1]),  ([0, 1], [1]),  ([0, 0], [0])]
xor = NeuralNet(2, 1, 1) # 2, 2, 1
xor.train(xor_data, iters=10000, print_interval=500)
print(xor.test_with_expected(xor_data))

voter_data = [([.9, .6, .8, .3, .1], [1.0]), ([.8, .8, .4, .6, .4], [1.0]), ([.7, .2, .3, .6, .3], [1.0]), ([.5, .5, .8, .4, .8], [0.0]), ([.3, .1, .6, .8, .8], [0.0]), ([.6, .3, .4, .3, .6], [0.0]),]
voter = NeuralNet(5, 3, 1)
voter.train(voter_data)
results = [
    voter.evaluate([1.0, 1.0, 1.0, .1, .1]),
    voter.evaluate([.5, .2, .1, .7, .7]),
    voter.evaluate([.8, .3, .3, .3, .8]),
    voter.evaluate([.8, .3, .3, .8, .3]),
    voter.evaluate([.9, .8, .8, .3, .6])
]
print(results)
