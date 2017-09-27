import ast
import time
import dill
from itertools import chain
import tensorflow as tf
import numpy as np
from layers import Input, Layer, layer_from_dicts
from utils import print_headers, print_current, print_metrics
import functions


def save_model(model, filepath):
    tensors = []
    layers = dict()
    params = dict()
    tensor_list = model.outputs if isinstance(model.outputs, list) else [model.outputs]
    while tensor_list:
        tensor = tensor_list.pop()
        if tensor.input is not None:
            tensor_list.append(tensor.input)
        if tensor.node is not None:
            layer = tensor.node
            if layer.name not in layers:
                layer_dict = layer.get_config()
                if layer.W is not None:
                    layer_dict.update({'W': layer.W.name})
                    params.update({layer.W.name: (layer.W.get_shape().as_list(), Model.session.run(layer.W))})
                if layer.b is not None:
                    layer_dict.update({'b': layer.b.name})
                    params.update({layer.b.name: (layer.b.get_shape().as_list(), Model.session.run(layer.b))})
                layers.update({layer.name: layer_dict})
        tensor_tuple = (
            tensor.name,
            tensor.input.name if tensor.input is not None else None,
            tensor.node.name if tensor.node is not None else 'Input.T.%s' % tensor.output.get_shape().as_list()[1:]
        )
        if tensor_tuple in tensors:
            tensors.remove(tensor)
        tensors = [tensor_tuple] + tensors

    inputs = [i.name for i in model.inputs] if isinstance(model.inputs, list) else model.inputs.name
    outputs = [i.name for i in model.outputs] if isinstance(model.outputs, list) else model.outputs.name
    model_dict = {
        'inputs': inputs,
        'outputs': outputs,
        'optimizer': model.optimizer,
        'loss': model.loss,
        'metrics': model.metrics,
        'tensors': tensors,
        'layers': layers,
        'params': params
    }

    dill.dump(model_dict, open(filepath, 'wb'))


def load_model(filepath):
    model_dict = dill.load(open(filepath, 'rb'))
    params = model_dict['params']
    layers = model_dict['layers']
    for name, layer_dict in layers.items():
        W_name = layer_dict.pop('W') if 'W' in layer_dict else None
        b_name = layer_dict.pop('b') if 'b' in layer_dict else None
        layer = layer_from_dicts(layer_dict)
        if W_name is not None:
            layer.W = Layer._weight_variable(params[W_name][0], layer.name)
            Model.session.run(layer.W.assign(params[W_name][1]))
        if b_name is not None:
            layer.W = Layer._bias_variable(params[b_name][0], layer.name)
            Model.session.run(layer.W.assign(params[b_name][1]))
        layers.update({name: layer})
    tensor_dict = dict()
    for tensor in model_dict['tensors']:
        tensor_name = tensor[0]
        layer_name = tensor[2]
        if 'Input.T.' in layer_name:
            tensor_dict.update({tensor_name: Input(ast.literal_eval(layer_name.replace('Input.T.', '')))})
        else:
            layer = layers[layer_name]
            input_tensor = tensor_dict[tensor[1]]
            tensor_dict.update({tensor_name: layer(input_tensor)})
    inputs = model_dict['inputs']
    inputs = [tensor_dict[i_name] for i_name in inputs] if isinstance(inputs, list) else tensor_dict[inputs]
    outputs = model_dict['outputs']
    outputs = [tensor_dict[i_name] for i_name in outputs] if isinstance(outputs, list) else tensor_dict[outputs]
    return Model(inputs, outputs, model_dict['optimizer'], model_dict['loss'], model_dict['metrics'])


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
    def initialise_vars():
        uninit_vars = [var for var in tf.global_variables() if not Model.session.run(tf.is_variable_initialized(var))]
        Model.session.run(tf.variables_initializer(uninit_vars))

    def __init__(self, inputs, outputs, optimizer, loss, metrics):
        # TODO: Multiple inputs/outputs
        # TODO: Checking if input is tensor if not create one for it
        self.inputs = inputs
        self.outputs = outputs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.best_params = dict()
        Model.initialise_vars()

    @property
    def layers(self):
        tensor_list = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        layers = []
        while tensor_list:
            tensor = tensor_list.pop()
            if tensor.input is not None:
                tensor_list.append(tensor.input)
            if tensor.node is not None:
                if tensor.node in layers:
                    layers.remove(tensor.node)
                layers = [tensor.node] + layers
        return layers

    @property
    def layers_dict(self):
        tensor_list = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        layers = dict()
        while tensor_list:
            tensor = tensor_list.pop()
            if tensor.input is not None:
                tensor_list.append(tensor.input)
            layer = tensor.node
            if layer is not None:
                if layer.name not in layers:
                    layers.update({layer.name: layer})
        return layers

    def _update_best_params(self):
        for l in self.layers:
            if l.trainable:
                best_W = Model.session.run(l.W) if l.W is not None else None
                best_b = Model.session.run(l.b) if l.b is not None else None
                self.best_params.update({l.name: (best_W, best_b)})

    def count_trainable_parameters(self):
        parameters = 0
        for l in self.layers:
            parameters += l.count_trainable_parameters()
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
        # TODO: Checking if input is tensor if not create one for it
        inputs = self.inputs.output
        outputs = tf.placeholder(tf.float32, self.outputs.output.get_shape().as_list())
        loss = Model.metric_functions[self.loss](self.outputs.output, outputs)
        metrics = Model.metric_functions[self.metrics](self.outputs.output, outputs)
        optimizer = Model.optimizers[self.optimizer].minimize(loss)

        n_batches = -(-len(tr_data) / batch_size)
        train_loss = {'train_loss': [np.inf, np.inf, 0]}
        train_acc = {'train_acc': [-np.inf, -np.inf, 0]}
        val_loss = {'val_loss': [np.inf, np.inf, 0]}
        val_acc = {'val_acc': [-np.inf, -np.inf, 0]}

        metrics_dict = dict(chain.from_iterable(map(dict.items, [train_loss, train_acc, val_loss, val_acc])))
        print_headers(train_loss, train_acc, val_loss, val_acc)
        t_start = time.time()
        no_improvement = 0

        Model.initialise_vars()
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
                Model.session.run(optimizer, feed_dict={inputs: batch_xs, outputs: batch_ys})
                curr_acc = Model.session.run(metrics, feed_dict={inputs: batch_xs, outputs: batch_ys})
                curr_loss = Model.session.run(loss, feed_dict={inputs: batch_xs, outputs: batch_ys})
                acc_sum += curr_acc
                loss_sum += curr_loss
                curr_values = (curr_loss, loss_sum / (step + 1), curr_acc, acc_sum / (step + 1))
                print_current(i, step, n_batches, curr_values)

            train_loss['train_loss'][1] = loss_sum / n_batches
            train_acc['train_acc'][1] = acc_sum / n_batches
            val_loss['val_loss'][1] = Model.session.run(loss, feed_dict={inputs: val_data, outputs: val_labels})
            val_acc['val_acc'][1] = Model.session.run(metrics, feed_dict={inputs: val_data, outputs: val_labels})
            print_metrics(i, train_loss, train_acc, val_loss, val_acc, time.time() - t_in)
            if metrics_dict[monitor][2] != i:
                no_improvement += 1
            else:
                self._update_best_params()
                no_improvement = 0
            if no_improvement >= patience:
                break
        t_end = time.time() - t_start
        print('Training finished in %d epochs (%fs) with %s = %f (epoch %d)' %
              (i, t_end, monitor, metrics_dict[monitor][0], metrics_dict[monitor][2]))
        for k, v in self.best_params.items():
            if self.layers_dict[k].W is not None:
                Model.session.run(self.layers_dict[k].W.assign(v[0]))
            if self.layers_dict[k].b is not None:
                Model.session.run(self.layers_dict[k].b.assign(v[1]))
