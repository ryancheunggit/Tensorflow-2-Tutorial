# Chapter4
## Optimizers

In this chapter, we will introduce how to implement different optimizers.  

### 0. Setups for this section
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
print(tf.__version__)
tf.random.set_seed(42)
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.boston_housing.load_data()
y_tr, y_te = map(lambda x: np.expand_dims(x, -1), (y_tr, y_te))
x_tr, y_tr, x_te, y_te = map(lambda x: tf.cast(x, tf.float32), (x_tr, y_tr, x_te, y_te))


class MLP(tf.keras.Model):
    def __init__(self, num_hidden_units, num_targets, hidden_activation='relu', **kwargs):
        super().__init__(**kwargs)
        if type(num_hidden_units) is int: num_hidden_units = [num_hidden_units]
        self.feature_extractor = tf.keras.Sequential([tf.keras.layers.Dense(unit, activation=hidden_activation)
                                                      for unit in num_hidden_units])
        self.last_linear = tf.keras.layers.Dense(num_targets, activation='linear')

    @tf.function
    def call(self, x):
        features = self.feature_extractor(x)
        outputs = self.last_linear(features)
        return outputs
```
> ```Console
> 2.1.0
> ```

### 1. Gradient Descent {#gd}
Lets formally look at the gradient descent algorithm we have been using so far.  
<div class='formula'>
1. $$g \leftarrow \nabla_{\theta}J(\theta)$$ <br>
2. $$u \leftarrow - \eta g$$ <br>
3. $$\theta \leftarrow  \theta + u$$
</div>  

Where:
+ $$g \leftarrow \nabla_{\theta}J(\theta)$$ is the gradient of loss function $$J$$ with respect to our model's parameters $$\theta$$.  
+ $$u$$ is the updates we will apply to each parameters, in this case it is the negative gradient multiplied by the learning rate $$\eta$$.  
+ Lastly, the update rule is to increment $$\theta$$ by $$u$$.  

Take a look at our previous optimization code:
```python
gradients = tape.gradient(loss, model.variables)

for g, v in zip(gradients, model.variables):  
    v.assign_add(tf.constant([-0.05], dtype=tf.float32) * g)
```  

With only a slight tweak, the code will become literally a line by line translation of the formula.
```python
eta = tf.constant(-.05, dtype=tf.float32)
gradients = tape.gradient(loss, model.variables)
for grad, var in zip(gradients, model.variables):  
    update = - eta * grad
    var.assign_add(update)
```

Its obvious that our optimization contains some data(the learning rate $$\eta$$) and some operations(the parameter updates). We can put them into a class.  
```python
class GradientDescent(object):
    def __init__(self, lr=.01):
        self._lr = tf.Variable(lr, dtype=tf.float32)

    def apply_gradients(self, grads_and_vars):
        for grad, var in grads_and_vars:
            update = - self._lr * grad
            var.assign_add(update)
```  

With this optimizer class, our training code is becoming more modularized.
```python
@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(y - model(x)))
    gradients = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(gradients, model.variables))    
    return loss


@tf.function
def test_step(model, x, y):
    return tf.reduce_mean(tf.square(y - model(x)))


def train(model, optimizer, n_epochs=10000, his_freq=100):
    history = []
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_step(model, optimizer, x_tr, y_tr)
        te_loss = test_step(model, x_te, y_te)
        if not epoch % his_freq:
            history.append({'epoch': epoch,
                            'training_loss': tr_loss.numpy(),
                            'testing_loss': te_loss.numpy()})
    return model, pd.DataFrame(history)
```

Recall that from the end of last chapter, our model is no better than a constant function. Wondering if learning rate of the optimizer has anything to do with it. Let's to code up some experiments on how learning rate affects the training progress.  
```python
def lr_experiments(learning_rates):
    experiments = []
    for lr in learning_rates:
        model, history = train(MLP(4, 1), GradientDescent(lr))
        history['lr'] = lr
        experiments.append(history)

    experiments = pd.concat(experiments, axis=0)
    return experiments

experiments = lr_experiments(learning_rates=[10 ** -i for i in range(3, 10)])
ax = experiments.\
    pivot(index='epoch', columns='lr', values='testing_loss').\
    plot(kind='line', logy=True, figsize=(12, 6))

ax.get_figure().savefig('ch4_plot_1.png')
print(experiments.groupby('lr')['testing_loss'].min())
```
> ```Console
> lr
> 1.000000e-09    216.570267
> 1.000000e-08    616.280029
> 1.000000e-07     85.039314
> 1.000000e-06     71.763756
> 1.000000e-05    444.852142
> 1.000000e-04     97.061386
> 1.000000e-03     83.243866
> Name: testing_loss, dtype: float64
> ```
> ![gradient descent lr experiment](/images/ch4_plot_1.png)    

We see that with learning rate 1e-6, we can finally beat the naive baseline model, which is a constant. Wait a second, in what situation would our network be equivalent to a constant? Only if the activations are mostly zeros, and thus the model would almost always output the bias term from the last layer. Let's see if that is the case.   
```python
model, history = train(MLP(4, 1), GradientDescent(1e-3))
print(model.layers[0](x_te))
print(model.layers[0](x_te) @ model.layers[1].variables[0])
print(model.layers[1].variables[1])
```
> ```Console
> tf.Tensor(
> [[0. 0. 0. 0.]
>  [0. 0. 0. 0.]
>  ...
>  [0. 0. 0. 0.]
>  [0. 0. 0. 0.]], shape=(102, 4), dtype=float32)
> tf.Tensor(
> [[0.]
>  [0.]
>  ...
>  [0.]
>  [0.]], shape=(102, 1), dtype=float32)
> <tf.Variable 'dense_15/bias:0' shape=(1,) dtype=float32, numpy=array([22.394573], dtype=float32)>
> ```

Indeed, with learning rate 1e-3 our network basically collapsed to a bias term. In this case, the initial gradients are too big, if the learning rate is also very big, it can kill almost all the units immediately. So, we would need to either reduce the learning rate or apply some kind of control of the gradient values. Normalize the gradient is a good addition we should add to the gradient descent optimizer.   
```python
class GradientDescent(object):
    def __init__(self, lr=.01, grad_norm=None):
        self._lr = tf.Variable(lr, dtype=tf.float32)
        self.grad_norm = grad_norm

    def apply_gradients(self, grads_and_vars):
        for grad, var in grads_and_vars:
            if self.grad_norm > 0:
                grad = tf.clip_by_norm(grad, self.grad_norm)
            update = - self._lr * grad
            var.assign_add(update)

model, history = train(MLP(4, 1), GradientDescent(1e-3, grad_norm=2))
ax = history.plot(x='epoch', logy=True, figsize=(12, 6))
ax.get_figure().savefig('ch4_plot_2.png')
print(history['testing_loss'].min())
```
```Console
36.491302490234375
```
> ![gradient descent with grad clip](/images/ch4_plot_2.png)    

Now, we trid training our model again, this time, we normalize the gradients to have l2 norm equal to a small number 2, and indeed we got much better result even with the previously problematic learning rate value 1e-3.  

### 2. Stochastic Gradient Descent {#sgd}
Yet another problem with the above gradient descent optimization is that it uses the full dataset every time it needs to do a parameter update. It can be very both very slow and memory hungry. A quick fix is to use small samples of the data to get estimates of the gradients and perform much more frequent updates. This is called stochastic gradient descent(SGD).

To train our model with SGD, we don't need to modify our optimizer class. Instead, we need to modify the training code to include sampling. To make life a bit easier, we can wrap the data tensors into iterators.
```python
class Dataset(object):
    def __init__(self, tensors, batch_size=32, shuffle=True):
        self.tensors = tensors if isinstance(tensors, (list, tuple)) else (tensors, )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total = tensors[0].shape[0]
        assert all(self.total == tensor.shape[0] for tensor in self.tensors), 'Tensors should have matched length'
        self.n_steps = self.total // self.batch_size
        self._indices = tf.range(self.total)

    def __iter__(self):
        self._i = 0
        if self.shuffle:
            self._indices = tf.random.shuffle(self._indices)
        return self

    def __next__(self):
        if self._i >= self.n_steps:
            raise StopIteration
        else:
            start = self._i * self.batch_size
            end = start + self.batch_size
            indices = self._indices[start: end]
            samples = (tf.gather(tensor, indices) for tensor in self.tensors)
            self._i += 1
            return samples
```
With this `Dataset` class, we can make very small change to the training code.
```python
train_dataset = Dataset((x_tr, y_tr))
test_dataset = Dataset((x_te, y_te))


def train(model, optimizer, n_epochs, batch_size=32, his_freq=10):
    history = []
    for epoch in range(1, n_epochs + 1):
        tr_loss = []
        for x, y in train_dataset:
            tr_loss.append(train_step(model, optimizer, x, y).numpy())

        te_loss = []
        for x, y in test_dataset:
            te_loss.append(test_step(model, x, y).numpy())

        te_loss_full = test_step(model, x_te, y_te)

        if not epoch % his_freq:
            history.append({'epoch': epoch,
                            'training_loss': np.mean(tr_loss),
                            'testing_loss': np.mean(te_loss),
                            'testing_loss_full': te_loss_full.numpy()})
    return model, pd.DataFrame(history)

model, history = train(MLP(4, 1), GradientDescent(1e-3, 2), n_epochs=2000)
ax = history.plot(x='epoch', kind='line', figsize=(12, 6))
ax.get_figure().savefig('ch4_plot_3.png')

print(history.testing_loss_full.min())
```
> ```Console
> 31.820463180541992
> ```
> ![stochastic gradient descent training loss](/images/ch4_plot_3.png)    

We can see that training with small batches of random samples, the trajectory of loss values becomes a bit zig-zag in shape, but still follows the similar path as previously when we train with full dataset.
