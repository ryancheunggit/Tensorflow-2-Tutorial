# Chapter4
## Optimizers

In this chapter, we will introduce how to implement different optimizers.  

### 0. Setups for this section
```python
import tensorflow as tf
from pprint import pprint
print(tf.__version__)
tf.random.set_seed(42)
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.boston_housing.load_data()
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
> 2.0.0
> ```

### 1. Gradient Descent {#gd}
Lets formally look at the gradient descent algorithm we have been using so far.  
<div class='formula'>$$
g \leftarrow \nabla_{\theta}J(\theta) \\
u \leftarrow - \eta g \\
\theta \leftarrow  \theta + u
$$</div>  

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


def train(model, optimizer, n_epochs=1000, his_freq=10):
    history = []
    for iteration in range(1, n_epochs + 1):
        tr_loss = train_step(model, optimizer, x_tr, y_tr)
        te_loss = test_step(model, x_te, y_te)
        if not iteration % his_freq:
            history.append({'iteration': iteration,
                            'training_loss': tr_loss.numpy(),
                            'testing_loss': te_loss.numpy()})
    return model, pd.DataFrame(history)
```

Let's see how this is allowing us to code up some experiments on how learning rate affects the training progress very quickly.  
```python
experiments = []
for lr in [10**-i for i in range(1, 5)]:
    model, history = train(MLP(4, 1), GradientDescent(lr))
    history['lr'] = lr
    experiments.append(history)

experiments = pd.concat(experiments, axis=0)
ax = experiments.pivot(index='iteration', columns='lr', values='training_loss').plot(kind='line', logy=True, figsize=(12,6))
fig = ax.get_figure()
fig.savefig('ch4_plot_1.png')
```
> ![gradient descent lr experiment](/images/ch4_plot_1.png)    

We see that with very few line of code we cooked up some experiments, which we can easily tweak to run experiment on different combinations of model, learning rate.  

### 2. Stochastic Gradient Descent {#sgd}
