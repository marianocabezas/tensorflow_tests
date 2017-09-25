import time
from itertools import chain
import numpy as np
import tensorflow as tf
from utils import print_headers, print_current, print_metrics
import functions


class Model(object):
    # TODO: Improve dictionary of "metrics"
    # That includes creating all the necessary functions and prepare the logic given some output layers:
    # Create a placeholder function with the shape of the output layers, pass them as "y" on the function and
    # pass the tensorflow layer as "x"
    # I should also include possible adversarial metrics. That means rework metric functions
    #  to check if the variables are tensors or not. If they aren't create a tensor for it.
    metric_functions = {
        'categorical_crossentropy': functions.cross_entropy,
        'accuracy': functions.accuracy
    }

    optimizers = {
        'adam': tf.train.AdamOptimizer(1e-4)
    }

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    @staticmethod
    def _get_layer_graph(inputs, outputs, graph=dict()):
        if not isinstance(inputs, list):
            inputs = [inputs]
        t_layer = outputs
        while t_layer not in inputs:
            graph.update({t_layer.name: t_layer})
            t_layer = t_layer.input
        return graph

    def __init__(self, inputs, outputs, optimizer, loss, metrics):
        # TODO: Compute layer dictionary + trainable variable dictionary
        self.layers = Model._get_layer_graph(inputs, outputs)
        # TODO: Multiple inputs/outputs
        # TODO: Checking if input is tensor if not create one for it
        self.inputs = inputs.output
        self.outputs = outputs.output
        self.sym_outputs = tf.placeholder(tf.float32, self.outputs.get_shape().as_list())
        self.loss = Model.metric_functions[loss](self.outputs, self.sym_outputs)
        self.optimizer = Model.optimizers[optimizer].minimize(self.loss)
        self.metrics = Model.metric_functions[metrics](self.outputs, self.sym_outputs)

    def count_trainable_parameters(self):
        parameters = 0
        for k, v in self.layers.items():
            parameters += v.count_trainable_parameters()
        return parameters

    def fit(
            self,
            tr_data,
            tr_labels,
            epochs,
            batch_size,
            val_data=None,
            val_labels=None,
            patience=np.inf,
            monitor='val_acc'
    ):
        # TODO: Adapt it to multiple inputs/outputs while also changing the names of multiple metrics and losses
        # Multiple inputs/outputs should be passed by name, which will probably be tricky
        Model.session.run(tf.global_variables_initializer())
        inputs = self.inputs
        outputs = self.sym_outputs
        n_batches = -(-len(tr_data) / batch_size)
        train_loss = {'train_loss': [np.inf, np.inf, 0]}
        train_acc = {'train_acc': [-np.inf, -np.inf, 0]}
        val_loss = {'val_loss': [np.inf, np.inf, 0]}
        val_acc = {'val_acc': [-np.inf, -np.inf, 0]}

        metrics = dict(chain.from_iterable(map(dict.items, [train_loss, train_acc, val_loss, val_acc])))
        print_headers(train_loss, train_acc, val_loss, val_acc)
        t_start = time.time()
        no_improvement = 0
        for i in range(epochs):
            idx = np.random.permutation(len(tr_data))
            x = tr_data[idx, :]
            y = tr_labels[idx, :]
            acc_sum = 0
            loss_sum = 0
            t_in = time.time()
            for step in range(n_batches):
                step_init = step * batch_size
                step_end = step * batch_size + batch_size
                batch_xs, batch_ys = x[step_init:step_end, :], y[step_init:step_end, :]
                Model.session.run(self.optimizer, feed_dict={inputs: batch_xs, outputs: batch_ys})
                curr_acc = Model.session.run(self.metrics, feed_dict={inputs: batch_xs, outputs: batch_ys})
                curr_loss = Model.session.run(self.loss, feed_dict={inputs: batch_xs, outputs: batch_ys})
                acc_sum += curr_acc
                loss_sum += curr_loss
                curr_values = (curr_loss, loss_sum / (step + 1), curr_acc, acc_sum / (step + 1))
                print_current(i, step, n_batches, curr_values)

            train_loss['train_loss'][1] = loss_sum / n_batches
            train_acc['train_acc'][1] = acc_sum / n_batches
            val_loss['val_loss'][1] = Model.session.run(self.loss, feed_dict={inputs: val_data, outputs: val_labels})
            val_acc['val_acc'][1] = Model.session.run(self.metrics, feed_dict={inputs: val_data, outputs: val_labels})
            print_metrics(i, train_loss, train_acc, val_loss, val_acc, time.time() - t_in)
            # TODO: save weights (layer dictionary is a must)
            if metrics[monitor][2] != i:
                no_improvement += 1
            else:
                no_improvement = 0
            if no_improvement >= patience:
                break
        t_end = time.time() - t_start
        print('Training finished in %d epochs (%fs) with %s = %f (epoch %d)' %
              (i, t_end, monitor, metrics[monitor][0], metrics[monitor][2]))
