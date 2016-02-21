import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import  rnn_test

words, wv = rnn_test.build_wv()
data, label, label_hv = rnn_test.get_data(words)
print len(data)

def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    print
    return h

def grad_desc(cost, theta):
    alpha = 0.01 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

x = T.dvector()
y = T.dvector()

theta1 = theano.shared(np.array(np.random.rand(51,10), dtype=theano.config.floatX)) # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(11,10), dtype=theano.config.floatX))
theta3 = theano.shared(np.array(np.random.rand(11,5), dtype=theano.config.floatX))

hid1 = layer(x, theta1)
hid2 = layer(hid1, theta2)
out1 = layer(hid2, theta3) #output layer

fc = T.sum((out1 - y)**2) #cost expression

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),
        (theta2, grad_desc(fc, theta2)),
        (theta3, grad_desc(fc, theta3))])
run_forward = theano.function(inputs=[x], outputs=out1)

#inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2)
#exp_y = np.array([1, 1, 0, 0])
#[inputs, exp_y, _] = gen.generatepyt(100)
#print exp_y


cur_cost = 0
for i in range(10000):
    for k in range(len(data)):
        v = wv[words[data[k]]]
        cur_cost = cost(v, label_hv[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

'''
[test_x, real_y, label] = gen.generatepyt(20)
correct = 0
for i in range(len(test_x)):
    res = run_forward(test_x[i])
    max = res[0]
    max_j = 0
    for j in range(1,3):
        if res[j] > max:
            max = res[j]
            max_j = j
    if label[i] == max_j:
        correct += 1

print float(correct)/len(test_x)
'''