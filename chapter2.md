# Chapter 2
## AutoGraph

In this chapter, we will introduce how to convert python function into executable Tensorflow graphs.  

### 0. Setups for this section
```python
import inspect
import time
import numpy as np
import tensorflow as tf
from pprint import pprint
print(tf.__version__)
tf.random.set_seed(42)
np.random.seed(42)

true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]  
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)
```
> output
> ```Console
> 2.0.0
> ```

### 1. AutoGraph {#autograph}
A computational graph contains two things computation and data. Tensorflow graph [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) is the computational graph made from operations as computation units and tensors as data units. Tensorflow has many optimizations around graphs, so executing in graph mode result in better utilization of distributed computing. Instead of writing complicated graph mode code, Tensorflow 2 provides a tool *AutoGraph* to automatically analyze and convert python code into graph code which can then be traced to create a graph with *Function*.   

The following is a python function we defined in chapter 1. It involves relatively simple math operations on tensors, so it is pretty much *graph ready*. Let's see if [`tf.autograph`](https://www.tensorflow.org/api_docs/python/tf/autograph) would do anything interesting to it.
```python
def f(a, b, power=2, d=3):
    return tf.pow(a, power) + d * b

converted_f = tf.autograph.to_graph(f)
print(inspect.getsource(converted_f))
```
> output
> ```Console
>     def tf__f(a, b, power=None, d=None):
>       do_return = False
>       retval_ = ag__.UndefinedReturnValue()
>       with ag__.FunctionScope('f', 'f_scope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as f_scope:
>         do_return = True
>         retval_ = f_scope.mark_return_value(ag__.converted_call(tf.pow, f_scope.callopts, (a, power), None, f_scope) + d * b)
>       do_return,
>       return ag__.retval(retval_)
> ```

The generated function is a regular python function. Even though it may look much more complicated than the original function, but most stuff is boilerplate code to handling details of function scopes and overloads function calls with [`converted_call`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/impl/api.py#L377). The line `retval_ = f_scope.mark_return_value(ag__.converted_call(tf.pow, f_scope.callopts, (a, power), None, f_scope) + d * b)` tells us that the generated code is performing the same computation.

Let's move on to another python function with a bit graph unfriendly construct.
```python
def cube(x):
    o = x
    for _ in range(2):
        o *= x
    return o

converted_cube = tf.autograph.to_graph(cube)
print(inspect.getsource(converted_cube))
```
> output
> ```Console
> def tf__cube(x):
>   do_return = False
>   retval_ = ag__.UndefinedReturnValue()
>   with ag__.FunctionScope('cube', 'cube_scope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as cube_scope:
>     o = x
>
>     def get_state():
>       return ()
>
>     def set_state(_):
>       pass
>
>     def loop_body(iterates, o):
>       _ = iterates
>       o *= x
>       return o,
>     o, = ag__.for_stmt(ag__.converted_call(range, cube_scope.callopts, (2,), None, cube_scope), None, loop_body, get_state, set_state, (o,), ('o',), ())
>     do_return = True
>     retval_ = cube_scope.mark_return_value(o)
>   do_return,
>   return ag__.retval(retval_)
> ```

We see that body of the `for` loop is converted into a `loop_body` function, which is invoked by `autograph.for_stmt`. This [`for_stmt`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/operators/control_flow.py) operation is, in a sense, *overloads* the `for` statement. The purpose of this transformation is to make the code into a functional style so that it can be executed in graph.

Let's look at yet another function, this time with a conditional statement.
```python
def g(x):
    if tf.reduce_any(x < 0):
        return tf.square(x)
    return x
converted_g = tf.autograph.to_graph(g)
print(inspect.getsource(converted_g))
```
> output
> ```Console
>    def tf__g(x):
>       do_return = False
>       retval_ = ag__.UndefinedReturnValue()
>       with ag__.FunctionScope('g', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
>
>         def get_state():
>           return ()
>
>         def set_state(_):
>           pass
>
>         def if_true():
>           do_return = True
>           retval_ = fscope.mark_return_value(ag__.converted_call(tf.square, (x,), None, fscope))
>           return retval_, do_return
>
>         def if_false():
>           do_return = True
>           retval_ = fscope.mark_return_value(x)
>           return retval_, do_return
>         cond = ag__.converted_call(tf.reduce_any, (x < 0,), None, fscope)
>         retval_, do_return = ag__.if_stmt(cond, if_true, if_false, get_state, set_state, ('retval_', 'do_return'), ())
>       do_return,
>       return ag__.retval(retval_)
> ```

Here we see the similar thing happened with the conditional execution, it also has been converted into a functional form by *overload* the `if` statement.  

The big idea about AutoGraph is that it translates the python code we wrote into a style that can be traced to create Tensorflow graphs. During the transformation, great efforts were made to rewrite the data-dependent conditionals and control flows, as these statements can't be straight forward overloaded by Tensorflow operations. There are a lot more about AutoGraph, the interested reader can refer to the [paper](https://arxiv.org/abs/1810.08061) for more details.  

### 2. Functions {#function}
Once the code is graph friendly, we can trace the operations in the code to create a graph. The created graph then gets wrapped up in a [`ConcreteFunction`](https://www.tensorflow.org/guide/concrete_function) object so that we can execute computations backed in graph mode with it. The details of how tracing works is omitted from here, due to my limited understanding of it.    

Lets walkthrough how these works. First we provide Tensorflow our *graph friendly code* to create a [`Function`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/def_function.py#L327) object.
```python
tf_func_f = tf.function(autograph=False)(f)
tf_func_g = tf.function(autograph=False)(converted_g)
tf_func_g2 = tf.function(autograph=True)(g)
print(tf_func_f.python_function is f)
print(tf_func_g.python_function is converted_g)
print(tf_func_g2.python_function is g)
```
> output
> ```Console
> True
> True
> True
> ```  

As of now, there is no graph created yet. We only attached our function to this Function object. Notice that we specified `autograph=False` when constructing the Function in the first two cases, because we know both `f` and `converted_g` are graph friendly. But good old `g` is not, so we need to turn on `autograph` for it. In fact,  `tf.function(autograph=False)(tf.autograph.to_graph(g))` is roughly equivlent to `tf.function(autograph=True)(g)`.   

Note that we can create Function with `tf.function(autograph=False)(g)` this will succeed without error. But we won't be able to create any graphs with this Function in the next step.  

The next step is to provide Tensorflow a signature, i.e. a *description of inputs*, allowing it to create a graph by tracing how the input tensors would flow through the operations in the graph and record the graph in a callable object.
```python
concrete_g = tf_func_g.get_concrete_function(x=tf.TensorSpec(shape=[3], dtype=tf.float32))
print(concrete_g)
```
> output
> ``` Console
> <tensorflow.python.eager.function.ConcreteFunction object at 0x7f7cf8736240>
> ```  

We can use this concrete function directly as if it is a operation shipped with Tensorflow, or we can call the Function object, which will look up the concrete function and use it. Either way, the computation will be executed with the created graph.
```python
pprint(concrete_g(tf.constant([-1, 1, -2], dtype=tf.float32)))
pprint(tf_func_g(tf.constant([-1, 1, -2], dtype=tf.float32)))
```
> output
> ``` Console
> <tf.Tensor: id=45, shape=(3,), dtype=float32, numpy=array([1., 1., 2.], dtype=float32)>
> <tf.Tensor: id=74, shape=(3,), dtype=float32, numpy=array([1., 1., 2.], dtype=float32)>
> ```    

The Function object is like a graph factory. When detailed input specifications were provided, it uses the graph code as receipt to create new graphs. When asked with an known specifications, it will dig up the graph in the storage and serve it. When called with an unknown signature, it will trigger the creation of the concrete function first. Let's try to make a bunch graphs.
```python
concrete_f = tf_func_f.get_concrete_function(a=tf.TensorSpec(shape=[1], dtype=tf.float32), b=tf.TensorSpec(shape=[1], dtype=tf.float32))
print(concrete_f)
pprint(concrete_f(tf.constant(1.), tf.constant(2.)))  
pprint(tf_func_f(1., 2.))
pprint(tf_func_f(a=tf.constant(1., dtype=tf.float32), b=2, power=2.))
pprint(tf_func_f(a=tf.constant(1., dtype=tf.float32), b=2., d=3))  
pprint(tf_func_f(a=tf.constant(1., dtype=tf.float32), b=2., d=3., power=3.))
```
> output
> ```Console
> <tensorflow.python.eager.function.ConcreteFunction object at 0x7fb8a40bf080>
> <tf.Tensor: id=90, shape=(), dtype=float32, numpy=7.0>
> <tf.Tensor: id=99, shape=(), dtype=float32, numpy=7.0>
> <tf.Tensor: id=109, shape=(), dtype=float32, numpy=7.0>
> <tf.Tensor: id=111, shape=(), dtype=float32, numpy=7.0>
> <tf.Tensor: id=121, shape=(), dtype=float32, numpy=7.0>
> ```

How many graphs we created during the block of code above? The answer is 4. Can you figure out which call created which graph?   
```python
print(tf_func_f._get_tracing_count())
```
> ```Console
> 4
> ```

```python
for i, f in enumerate(tf_func_f._list_all_concrete_functions_for_serialization()):
    print(i, f.structured_input_signature)
```
> output
```Console
> 0 ((TensorSpec(shape=(), dtype=tf.float32, name='a'), 2.0, 3.0, 3.0), {})
> 1 ((TensorSpec(shape=(1,), dtype=tf.float32, name='a'), TensorSpec(shape=(1,), dtype=tf.float32, name='b'), 2, 3), {})
> 2 ((1.0, 2.0, 2, 3), {})
> 3 ((TensorSpec(shape=(), dtype=tf.float32, name='a'), 2, 2.0, 3), {})
```  

We can see that with slight difference in the input signature, Tensorflow would create multiple graphs, which may ended up locking a lot of resources if not managed right.  

Lastly, `tf.function` is also available as a decorator, which makes life easier. The following two are equivalent.
```python
@tf.function(autograph=False)
def square(x):
    return x * x

def square(x):
    return x * x
square = tf.function(autograph=False)(square)
```

### 3. Linear Regression Revisited {#linear}
Now let's go back to our linear regression example from last time and try to sugar it up with `tf.function`. Let's run the baseline.
```python
t0 = time.time()
for iteration in range(1001):
    with tf.GradientTape() as tape:
        y_hat = tf.linalg.matmul(x, weights)
        loss = tf.reduce_mean(tf.square(y - y_hat))

    if not (iteration % 200):
        print('mean squared loss at iteration {:4d} is {:5.4f}'.format(iteration, loss))

    gradients = tape.gradient(loss, weights)
    weights.assign_add(-0.05 * gradients)

pprint(weights)

print('time took: {} seconds'.format(time.time() - t0))
```
> output
> ```Console
> mean squared loss at iteration    0 is 16.9626
> mean squared loss at iteration  200 is 0.0363
> mean squared loss at iteration  400 is 0.0062
> mean squared loss at iteration  600 is 0.0012
> mean squared loss at iteration  800 is 0.0003
> mean squared loss at iteration 1000 is 0.0001
> <tf.Variable 'Variable:0' shape=(5, 1) dtype=float32, numpy=
> array([[-1.9201761e-03],
>        [ 1.0043812e+00],
>        [ 2.0000753e+00],
>        [ 3.0021193e+00],
>        [ 3.9955370e+00]], dtype=float32)>
> time took: 1.3443846702575684  seconds
```

Let's see if `@tf.function` can speed things up.
```python
t0 = time.time()

weights = tf.Variable(tf.random.uniform((5, 1)), dtype=tf.float32)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        y_hat = tf.linalg.matmul(x, weights)
        loss = tf.reduce_mean(tf.square(y - y_hat))
    gradients = tape.gradient(loss, weights)
    weights.assign_add(-0.05 * gradients)
    return loss

for iteration in range(1001):
    loss = train_step()
    if not (iteration % 200):
        print('mean squared loss at iteration {:4d} is {:5.4f}'.format(iteration, loss))

pprint(weights)
print('time took: {} seconds'.format(time.time() - t0))
```
> output
> ```Console
> mean squared loss at iteration    0 is 18.7201
> mean squared loss at iteration  200 is 0.0325
> mean squared loss at iteration  400 is 0.0017
> mean squared loss at iteration  600 is 0.0002
> mean squared loss at iteration  800 is 0.0000
> mean squared loss at iteration 1000 is 0.0000
> <tf.Variable 'Variable:0' shape=(5, 1) dtype=float32, numpy=
> array([[-3.1365452e-03],
>        [ 1.0065703e+00],
>        [ 2.0000944e+00],
>        [ 3.0032609e+00],
>        [ 3.9934685e+00]], dtype=float32)>
> time took: 0.4259765148162842 seconds
> ```

Even as simple as our example is, we can get some nice speed up with computing in graph mode.   

### 4. Caveats {#caveats}
To Be Con'd

### [Appendix. Code for this Chapter](https://gist.github.com/ryancheunggit/37cf2c84473168c60a3ea64445907e27)
