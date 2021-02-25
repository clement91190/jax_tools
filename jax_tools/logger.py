class BaseLogger:
    def __init__(self, log_level='INFO', **kwargs):
        pass

    def log(self):
        raise NotImplementedError(" Implement in Child classes ")


class EveryXIterCallbackLogger(BaseLogger):
    """ every n_iter calls of the log function the
    callback is called."""
    def __init__(self, callback=print, n_iter=10, **kwargs):
        BaseLogger.__init__(self, **kwargs)
        self.counter = 0
        self.n_iter = n_iter
        self.callback = callback

    def log(self, **kwargs):
        """ Every n_iter iteration callback is called with the params """ 
        self.counter += 1
        if self.counter % self.n_iter == 0:
            self.callback(**kwargs)

    def display_accuracy(self, msg):
        print(msg)


class TensorBoardLogger:
    """
    Logger for Tensorboard """
    #TODO
    pass