######################
# 模型训练
######################
from tensorflow.examples.tutorials.mnist import input_data
from backward import backward

def start():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    backward(mnist)

if __name__ == "__main__":
    start()