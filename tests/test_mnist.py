base_path='/Users/sivaramanks/code/PyNN/'
import sys
sys.path.append(base_path)
from datasets import mnist

from time import time

x_train, y_train = mnist.load_train_data(one_hot_label=False, flatten_input=True)
x_test, y_test = mnist.load_test_data(one_hot_label=True, flatten_input=False)

print x_train.shape, y_train.shape
print x_test.shape, y_test.shape


