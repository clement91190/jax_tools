from jax_tools.environment import LOG_DIR
import haiku as hk
import jax
from jax_tools.logger import EveryXIterLogger


BASE_LOGGER = EveryXIterLogger(log_level='INFO')

class BaseModel():
    def __init__(self, logger=BASE_LOGGER, prng_key=42, log_dir=LOG_DIR):
        self.compile()
        self.prng_key = jax.random.PRNGKey(prng_key)
        self.logger = logger

    def net_fn(self):
        """ function representing the neural network """

        raise NotImplementedError("Implement in children classes ")

    def loss(self):
        raise NotImplementedError("Implement in children classes ")

    def define_optimizer(self):
        """ good place to define your optimizer """
        pass 

    def set_optimizer(self, optimizer):
        """ Optimizer type from Optax, 
        should implement update and apply_updates and init functions """ 
        self.optimizer = optimizer

    def set_data_iterator(self, data_iterator):
        """data iterator to be used during the training loop """
        self.data_iterator = data_iterator

    def update(self, minibatch):
        """ Should also be implemented in children classes """
        raise NotImplementedError("Implement in children")
        #loss, grads = jax.value_and_grad(loss_fn)(params, images, labels)
        #self.logger.log()

    def init_optimizer(self):
        """ Must be called after init of network so that self.params is defined"""
        self.opt_state = self.optimizer.init(self.params)

    def loop_preprocess(self):
        pass

    def train(self, steps=100, *args, **kwargs):
 
        self.params = self.network.init(self.prng_key, next(self.data_iterator))
        if self.optimizer is not None:
            self.init_optimizer() 
        for _ in range(steps):
            minibatch = next(self.data_iterator)
            self.update(minibatch)

    def compile(self):
        net_fn_t = hk.transform(self.net_fn)
        self.network = hk.without_apply_rng(net_fn_t)

