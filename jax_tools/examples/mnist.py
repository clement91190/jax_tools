import haiku as hk
from jax_tools.base_model import BaseModel
from jax_toos.loss import softmax_cross_entropy
from jax_tools.logger import EveryXIterCallbackLogger
from jax_tools.updates import ema_update
import optax
from jax import numpy as jnp
import jax
from itertools import cycle


# logging function
def result_on_test_set(net, net_params, datasets):
    # Test on test set : 
    it_test = datasets['test'].as_numpy_iterator()

    def error_rate(y, y_hat):
        return jnp.mean(y != y_hat)

    errs = []
    for o in it_test:
        images, labels = o['image'].astype(jnp.float32) / 255., o['label']
        label_hat = net.apply(net_params, images).argmax(axis=1)
        errs.append(error_rate(labels, label_hat))
    print("Error on test set ", jnp.mean(errs))


class MnistBasicModel(BaseModel):
    def __init__(self):
        logger = EveryXIterCallbackLogger(n_iter=10, callback=result_on_test_set)
        BaseModel.__init__(self, logger=logger)
    
    def net_fn(self, images):
        # LNET 300 100 10
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(300),
            hk.Linear(100),
            hk.Linear(10)])
        return mlp(images)

    def loss_fn(self, params, images, labels):
        logits = self.net_fn_t.apply(params, images)
        return jnp.mean(softmax_cross_entropy(logits, labels))

    def loop_preprocess(self):
        self.avg_params = self.params

    def update(self, minibatch):
        images, labels = minibatch['image'].astype(jnp.float32) / 255., minibatch['label']
        grads = jax.grad(self.loss_fn)(self.params, images, labels)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        params = optax.apply_updates(self.params, updates)
        self.avg_params = ema_update(self.avg_params, params)
        self.log(net=self.network, net_params=params, datasets=self.datasets)

    def define_optimizer(self):
        self.optimizer = optax.adam(learning_rate=1e-3)


    def set_data_iterator(self, datasets):
        self.data_iterator = cycle(datasets['train'].as_numpy_iterator())   
        self.datasets = datasets

if __name__ == "__main__":
    # Fetch the dataset directly
    mnist = tfds.image.MNIST()
    mnist.download_and_prepare()
    datasets = mnist.as_dataset(batch_size=1000)


    model = MnistBasicModel()
    model.set_data_iterator(datasets) 
    model.train()