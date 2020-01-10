# Chapter1
## Tensors, Operations, Variables and Automatic Differentiation

In this chapter, we will introduce the bare minimum lower level Tensorflow APIs to get started building and training models.  

### 0. Setups for this section
```python
import numpy as np
import tensorflow as tf
from pprint import pprint
print(tf.__version__)
tf.random.set_seed(42)
```
> output
> ```Console
> 2.0.0
> ```

<a name="tensors"></a>
###  1. Tensors
Tensors are the objects that flow through operations(i.e., they are the inputs and the same time outputs of operations). They are N-dimensional arrays that hold elements of the same data type. The tensor objects in Tensorflow all have the shape and dtype properties, where shape is the number of elements that the tensor houses in each dimension, and dtype is the data type that all the elements within the tensor belong to.  

Let's create some tensors from python and numpy objects using [`tf.constant`](https://www.tensorflow.org/api_docs/python/tf/constant) and see their shapes and dtypes.
```python
scalar = tf.constant(1, dtype=tf.int8)
vector = tf.constant([1, 2, 3], dtype=tf.float32)
matrix = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.float32)
pprint([scalar, vector, matrix])
```  
> output
> ```Console
> [<tf.Tensor: id=0, shape=(), dtype=int8, numpy=1>,
>  <tf.Tensor: id=1, shape=(3,), dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>,
>  <tf.Tensor: id=2, shape=(2, 2), dtype=float32, numpy=
> array([[1., 2.],
>        [3., 4.]], dtype=float32)>]
> ```  

Under the hood, `__repr__` is using the `shape` accessor to get the shape information of the tensors. We can also use the [`tf.shape`](https://www.tensorflow.org/api_docs/python/tf/shape) operation to get the shape of a tensor object.  

```python
print(matrix.shape)
pprint(tf.shape(matrix))
```    
> output
> ```Console
> (2, 2)
> <tf.Tensor: id=3, shape=(2,), dtype=int32, numpy=array([2, 2], dtype=int32)>
> ```  

As for data type, we can't change a tensors' dtype with a mutator method as we can with numpy ndarrys(like `a = np.array([1, 2, 3]); a.dtype = np.float32)`). We can only use the [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/dtypes/cast) operation to create a new tensor with the desired new data type.
```python
matrix = tf.cast(matrix, dtype=tf.int8)
pprint(matrix)
```  
> output
> ```Console
> <tf.Tensor: id=4, shape=(2, 2), dtype=int8, numpy=
> array([[1, 2],
>        [3, 4]], dtype=int8)>
> ```

Notice the new tensor has the same values as the original tensor, but with int8 data type and it also has a different id number 4, indicating that it is a new tensor object.  

There are handy ways to create special tensors in Tensorflow, let's just sample a few of them.
```python
o = tf.zeros((2, 2))
x = tf.random.uniform((3, 2))
pprint(o)
pprint(x)
pprint(tf.ones_like(o))
```    
> output
> ```Console
> <tf.Tensor: id=7, shape=(2, 2), dtype=float32, numpy=
> array([[0., 0.],
>        [0., 0.]], dtype=float32)>
> <tf.Tensor: id=15, shape=(3, 2), dtype=float32, numpy=
> array([[0.6645621 , 0.44100678],
>        [0.3528825 , 0.46448255],
>        [0.03366041, 0.68467236]], dtype=float32)>
> <tf.Tensor: id=18, shape=(2, 2), dtype=float32, numpy=
> array([[1., 1.],
>        [1., 1.]], dtype=float32)>
> ```

We saw earlier that we can convert numpy arrays to Tensors with [`tf.constant`](https://www.tensorflow.org/api_docs/python/tf/constant), we can do the reverse with `.numpy()` method.
```python
x_numpy = x.numpy()
print(type(x_numpy))
```
> output
> ```Console
> <class 'numpy.ndarray'>
> ```

We can also explicitly copy tensors to devices.  
```python
with tf.device('/cpu:0'):
    x_cpu = tf.identity(x)
with tf.device('/gpu:0'):
    x_gpu = tf.identity(x)
print(x_cpu.device)
print(x_gpu.device)
```
> output
> ```Console
> /job:localhost/replica:0/task:0/device:CPU:0
> /job:localhost/replica:0/task:0/device:GPU:0
> ```

*Note we can actually do `x = x.cpu('0')` or `x = x.gpu('0')` to achieve the same, but these two are deprecating.*  

<a name="operations"></a>
### 2. Operations
Operations takes in tensors and computes output tensors. Both [`tf.shape`](https://www.tensorflow.org/api_docs/python/tf/shape) and [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/dtypes/cast) we saw earlier are actually tensor operations. Tensorflow provides a rich set of operations around tensors and these operations have routines for gradient computing built in.

Basic math operators in Python are overloaded by corresponding tensor operations. For example:
+ Addition
```python
e = tf.random.uniform((3, 2), dtype=x.dtype)
pprint(tf.math.reduce_all(x.__add__(e) == tf.add(x, e)))
```
> output
> ```Console
> <tf.Tensor: id=31, shape=(), dtype=bool, numpy=True>
> ```  

+ Element-wise multiplication
```python
pprint(tf.math.reduce_all(x.__mul__(e) == tf.multiply(x, e)))
```
> output
> ```Console
> <tf.Tensor: id=36, shape=(), dtype=bool, numpy=True>
> ```

The [`tf.linalg`](https://www.tensorflow.org/api_docs/python/tf/linalg) module contains linear algebra operations. For example:   
+ Matrix Multiplication
```python
pprint(tf.math.reduce_all(x.__matmul__(tf.transpose(e)) == tf.linalg.matmul(x, e, transpose_b=True)))
```
> output
> ```Console
> <tf.Tensor: id=43, shape=(), dtype=bool, numpy=True>
> ```

*TODO: work on a list of commonly used Tensorflow operations with example usecases.*

Slicing and indexing are operations implemented in the `__getitem__` method of [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) and the behavior is similar to numpy.
```python
pprint(matrix[0, 1])
pprint(matrix[1, :2])
pprint(matrix[tf.newaxis, 1, :2])
```
> output
> ```Console
> <tf.Tensor: id=47, shape=(), dtype=int8, numpy=2>
> <tf.Tensor: id=51, shape=(2,), dtype=int8, numpy=array([3, 4], dtype=int8)>
> <tf.Tensor: id=55, shape=(1, 2), dtype=int8, numpy=array([[3, 4]], dtype=int8)>
> ```

<a name="variables"></a>
### 3. Variables
Tensors are immutable, the values in tensors can't be updated in-place. Variables are just like Tensors in terms of as inputs to operations, but with the in-place value update methods.  

We can create variables with [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable).
```python
v = tf.Variable(x)
pprint(v)
```
> output
> ```Console
> <tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
> array([[0.6645621 , 0.44100678],
>        [0.3528825 , 0.46448255],
>        [0.03366041, 0.68467236]], dtype=float32)>
> ```

Variables can be inputs to operations just as Tensors, note the output is a Tensor not Variable.
```python
pprint(tf.square(v))
```
> output
> ```Console
> <tf.Tensor: id=65, shape=(3, 2), dtype=float32, numpy=
> array([[0.4416428 , 0.19448698],
>        [0.12452606, 0.21574403],
>        [0.00113302, 0.46877623]], dtype=float32)>
> ```

Variables can be updated with `.assign`, `.assign_add` or `.assign_sub` methods.
```python
v.assign(tf.square(v))
pprint(v)
```
> output
> ```Console
> <tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
> array([[0.4416428 , 0.19448698],
>        [0.12452606, 0.21574403],
>        [0.00113302, 0.46877623]], dtype=float32)>
> ```

```python
v.assign_sub(1 * tf.ones_like(v, dtype=v.dtype))
pprint(v)
```
> output
> ```Console
> <tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
> array([[-0.55835724, -0.805513  ],
>        [-0.8754739 , -0.784256  ],
>        [-0.998867  , -0.5312238 ]], dtype=float32)>
> ```  

Variables are typically used to represent the parameters and the states of the model, whereas the inputs, intermediate results, and outputs are Tensors.  

<a name="autodiff"></a>
### 4. Automatic Differentiation
Automatic differentiation is the implementation of the backpropagation algorithm which allows us to compute gradients of the loss function with respect to the model's parameters through chain rule. To do so, we need to keep track of the tensors and operations along the way. In Tensorflow we use the [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape) context to trace(recording/taping) what's happened inside, so that we can calculate the gradients afterward with the `.gradient()` from the tape object.
```python
a = tf.Variable([4], dtype=tf.float32)
b = tf.Variable([5], dtype=tf.float32)

def f(a, b, power=2, d=3):
    return tf.pow(a, power) + d * b

with tf.GradientTape(watch_accessed_variables=True) as tape:
    c = f(a, b)

pprint(tape.gradient(target=c, sources=[a, b]))
```
> output
> ```Console
> [<tf.Tensor: id=106, shape=(1,), dtype=float32, numpy=array([8.], dtype=float32)>,
>  <tf.Tensor: id=113, shape=(1,), dtype=float32, numpy=array([3.], dtype=float32)>]
> ```

By default, the context will only keep track of the variables but not the tensors, meaning that we can only ask for gradients with respect to the variables by default (through the `watch_accessed_variables` argument which defaults to `True`). But we can explicitly ask the tape to watch things for us.
```python
d = tf.constant(3, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(d)
    c = f(a, b, d=d)

pprint(tape.gradient(target=c, sources=[d]))
```
> output
> ```Console
> [<tf.Tensor: id=135, shape=(), dtype=float32, numpy=5.0>]
> ```

Note that once we extracted the gradients from the tape, the resources it holds will be released.
```python
with tf.GradientTape() as tape:
    c = f(a, b)

pprint(tape.gradient(c, [a]))
pprint(tape.gradient(c, [b]))
```
> output
> ```Console
> [<tf.Tensor: id=153, shape=(1,), dtype=float32, numpy=array([8.], dtype=float32)>]
> ---------------------------------------------------------------------------
> RuntimeError                              Traceback (most recent call last)
> <ipython-input-18-3b2edd8b0b33> in <module>
>       3
>       4 pprint(tape.gradient(c, [a]))
> ----> 5 pprint(tape.gradient(c, [b]))
> ...
> RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes.
> ```  

We can create a persistent gradient tape object to compute multiple gradients and manually release the resources.
```python
with tf.GradientTape(persistent=True) as tape:
    c = f(a, b)

pprint(tape.gradient(c, [a]))
pprint(tape.gradient(c, [b]))

del tape
```
> output
> ```Console
> [<tf.Tensor: id=175, shape=(1,), dtype=float32, numpy=array([8.], dtype=float32)>]
> [<tf.Tensor: id=197, shape=(1,), dtype=float32, numpy=array([3.], dtype=float32)>]
> ```

<a name="linear"></a>
### 5. Linear Regression
With tensors, variables, operations and automatic differentiation, we can start building models. Let's train a linear regression model with the gradient descent algorithm.  
```python
# ground truth
true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]  

# some random training data
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)

# model parameters
weights = tf.Variable(tf.random.uniform((5, 1)), dtype=tf.float32)

for iteration in range(1001):
    with tf.GradientTape() as tape:
        y_hat = tf.linalg.matmul(x, weights)
        loss = tf.reduce_mean(tf.square(y - y_hat))

    if not (iteration % 100):
        print('mean squared loss at iteration {:4d} is {:5.4f}'.format(iteration, loss))

    gradients = tape.gradient(loss, weights)
    weights.assign_add(-0.05 * gradients)

pprint(weights)
```
> output
> ```Console
> mean squared loss at iteration    0 is 15.1603
> mean squared loss at iteration  100 is 0.2243
> mean squared loss at iteration  200 is 0.0586
> mean squared loss at iteration  300 is 0.0161
> mean squared loss at iteration  400 is 0.0046
> mean squared loss at iteration  500 is 0.0013
> mean squared loss at iteration  600 is 0.0004
> mean squared loss at iteration  700 is 0.0001
> mean squared loss at iteration  800 is 0.0000
> mean squared loss at iteration  900 is 0.0000
> mean squared loss at iteration 1000 is 0.0000
> <tf.Variable 'Variable:0' shape=(5, 1) dtype=float32, numpy=
> array([[3.1751457e-03],
>        [1.0003914e+00],
>        [2.0022070e+00],
>        [2.9992850e+00],
>        [3.9944589e+00]], dtype=float32)>
> ```

We will use this linear regression training example in the next few chapters and gradually replacing parts of it with higher-level Tensorflow API equivalents.  

### [Appendix. Code for this Chapter](https://gist.github.com/ryancheunggit/37cf2c84473168c60a3ea64445907e27)
