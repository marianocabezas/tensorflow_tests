import ast
import time
import dill
from itertools import chain
import tensorflow as tf
import numpy as np
from layers import Input, Layer, layer_from_dicts
from utils import print_headers, print_current, print_metrics, train_test_split
import functions


def save_model(model, filepath):
    tensors = []
    layers = dict()
    params = dict()

    def get_model_config(tensor):
        if tensor.input is not None:
            if isinstance(tensor.input, list):
                tensor_input_name = [tensor_i.name for tensor_i in tensor.input]
            else:
                tensor_input_name = tensor.input.name
        else:
            tensor_input_name = None
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
            tensor_input_name,
            tensor.node.name if tensor.node is not None else 'Input.T.%s' % tensor.output.get_shape().as_list()[1:]
        )
        if tensor_tuple in tensors:
            tensors.remove(tensor_tuple)
        tensors.insert(0, tensor_tuple)

    model.iterate_tensor_graph(get_model_config)

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
            layer.b = Layer._bias_variable(params[b_name][0], layer.name)
            Model.session.run(layer.b.assign(params[b_name][1]))
        layers.update({name: layer})
    tensor_dict = dict()
    for tensor in model_dict['tensors']:
        tensor_name = tensor[0]
        layer_name = tensor[2]
        if 'Input.T.' in layer_name:
            tensor_dict.update({tensor_name: Input(ast.literal_eval(layer_name.replace('Input.T.', '')))})
        else:
            layer = layers[layer_name]
            input_tensor_name = tensor[1]
            input_tensor = tensor_dict[input_tensor_name] if not isinstance(input_tensor_name, list) else\
                [tensor_dict[i_name] for i_name in input_tensor_name]
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

    @staticmethod
    def _to_tf_tensor(tensor):
        return tf.placeholder(tf.float32, tensor.output.get_shape().as_list())

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
        layer_list = []

        def list_layers(tensor):
            if tensor.node is not None:
                if tensor.node in layer_list:
                    layer_list.remove(tensor.node)
                layer_list.insert(0, tensor.node)

        self.iterate_tensor_graph(list_layers)
        return layer_list

    @property
    def layers_dict(self):
        layers = dict()
        for layer in self.layers:
            layers.update({layer.name: layer})
        return layers

    def iterate_tensor_graph(self, f):
        tensor_list = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        while tensor_list:
            tensor = tensor_list.pop()
            if tensor.input is not None:
                if isinstance(tensor.input, list):
                    tensor_list += tensor.input
                else:
                    tensor_list.append(tensor.input)
            f(tensor)

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
            validation_split=0.25,
            patience=np.inf,
            monitor='val_acc'
    ):
        # TODO: Adapt it to multiple inputs/outputs while also changing the names of multiple metrics and losses
        # TODO: Checking if input is tensor if not create one for it
        # The multiple input/output stuff is finicky. In order to allow for multiple tensors, we assure that
        # both inputs and outputs are lists of either one or multiple tensors.
        # For inputs that's just ok as it is as long as we remember to do the same for the training/validation data
        # when creating the dictionaries we'll feed to tensorflow.
        # For outputs we have to do extra stuff to ensure one final unique metric to optimise. This is just a sum
        # of the loss functions for each output.
        # I might have to change stuff, but for now it's workable.
        model_outputs = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        tensor_inputs = [i.output for i in self.inputs] if isinstance(self.inputs, list) else [self.inputs.output]
        tensor_outputs = [Model._to_tf_tensor(o_i) for o_i in model_outputs]

        # Metrics/loss creation and optimizer. We also initialize the new variables created.
        loss_f = Model.metric_functions[self.loss]
        loss = tf.add_n([loss_f(o_i_a.output, o_i_gt) for o_i_a, o_i_gt in zip(model_outputs, tensor_outputs)])
        metric_f = Model.metric_functions[self.metrics]
        metrics = tf.add_n([metric_f(o_i_a.output, o_i_gt) for o_i_a, o_i_gt in zip(model_outputs, tensor_outputs)])
        optimizer = Model.optimizers[self.optimizer].minimize(loss)
        Model.initialise_vars()

        # Metrics/loss stuff for monitoring
        train_loss = {'train_loss': [np.inf, np.inf, 0]}
        train_acc = {'train_acc': [-np.inf, -np.inf, 0]}
        val_loss = {'val_loss': [np.inf, np.inf, 0]}
        val_acc = {'val_acc': [-np.inf, -np.inf, 0]}

        metrics_dict = dict(chain.from_iterable(map(dict.items, [train_loss, train_acc, val_loss, val_acc])))

        # DATA and TENSORS preprocessing. We ensure that everything is a list for easier control.
        tr_data = tr_data if isinstance(tr_data, list) else [tr_data]
        tr_labels = tr_labels if isinstance(tr_labels, list) else [tr_labels]
        if val_data is None:
            tr_x, tr_y, val_x, val_y = [
                train_test_split(tr_data_i, tr_labels_i, validation_split, np.random.random())
                for tr_data_i, tr_labels_i in zip(tr_data, tr_labels)
            ]
        else:
            tr_x = tr_data
            tr_y = tr_labels
            val_x = val_data if isinstance(val_data, list) else [val_data]
            val_y = val_labels if isinstance(val_labels, list) else [val_labels]
        tensors = tensor_inputs + tensor_outputs
        data = val_x + val_y
        val_feed_dict = dict((t_i, v_i) for t_i, v_i in zip(tensors, data))

        # Preloop stuff
        n_batches = -(-len(tr_x[0]) / batch_size)
        no_improvement = 0
        print_headers(train_loss, train_acc, val_loss, val_acc)

        # General timing
        t_start = time.time()

        for i in range(epochs):
            # Shuffle training data and prepare the variables to compute average loss/metric
            idx = [np.random.permutation(len(tr_data_i)) for tr_data_i in tr_x]
            x = [tr_data_i[idx_i, :] for tr_data_i, idx_i in zip(tr_x, idx)]
            y = [tr_labels_i[idx_i, :] for tr_labels_i, idx_i in zip(tr_y, idx)]
            acc_sum = 0
            loss_sum = 0

            # Epoch timing
            t_in = time.time()

            for step in range(n_batches):

                # Prepare the data dictionary for tensorflow
                step_init = step * batch_size
                step_end = step * batch_size + batch_size
                data = x + y
                tr_feed_dict = dict((t_i, v_i[step_init:step_end, :]) for t_i, v_i in zip(tensors, data))

                # Compute gradients, backpropagation and update weights using the optimizer
                Model.session.run(optimizer, feed_dict=tr_feed_dict)

                # Compute batch accuracy and loss and add it for the mean computation.
                # For "debugging" reasons we compute the average loss/metric for each step (that way
                # we can see the evolution per batch).
                curr_acc = Model.session.run(metrics, feed_dict=tr_feed_dict)
                curr_loss = Model.session.run(loss, feed_dict=tr_feed_dict)
                acc_sum += curr_acc
                loss_sum += curr_loss
                curr_values = (curr_loss, loss_sum / (step + 1), curr_acc, acc_sum / (step + 1))
                print_current(i, step, n_batches, curr_values)

            # Epoch loss/metric computation
            train_loss['train_loss'][1] = loss_sum / n_batches
            train_acc['train_acc'][1] = acc_sum / n_batches
            val_loss['val_loss'][1] = Model.session.run(loss, feed_dict=val_feed_dict)
            val_acc['val_acc'][1] = Model.session.run(metrics, feed_dict=val_feed_dict)
            print_metrics(i, train_loss, train_acc, val_loss, val_acc, time.time() - t_in)

            # We check if there was improvement and update the best parameters accordingly. Also, if patience is
            # specified we might apply early stopping.
            # We are enforcing a monitoring on a metric or loss (validation accuracy by default).
            if metrics_dict[monitor][2] != i:
                no_improvement += 1
            else:
                self._update_best_params()
                no_improvement = 0
            if no_improvement >= patience:
                break
        t_end = time.time() - t_start
        print('Training finished in %d epochs (%fs) with %s = %f (epoch %d)' %
              (i+1, t_end, monitor, metrics_dict[monitor][0], metrics_dict[monitor][2]))

        # Remember to update the best parameters
        for k, v in self.best_params.items():
            if self.layers_dict[k].W is not None:
                Model.session.run(self.layers_dict[k].W.assign(v[0]))
            if self.layers_dict[k].b is not None:
                Model.session.run(self.layers_dict[k].b.assign(v[1]))

    def predict(self, data, batch_size=32):
        # DATA preparation
        model_outputs = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        tensors = [i.output for i in self.inputs] if isinstance(self.inputs, list) else [self.inputs.output]

        # Preloop stuff
        data = data if isinstance(data, list) else [data]
        n_batches = -(-len(data[0]) / batch_size)

        outputs = []

        for step in range(n_batches):
            # Prepare the data dictionary for tensorflow
            step_init = step * batch_size
            step_end = step * batch_size + batch_size
            feed_dict = dict((t_i, v_i[step_init:step_end, :]) for t_i, v_i in zip(tensors, data))
            outputs.append([Model.session.run(output_i.output, feed_dict=feed_dict) for output_i in model_outputs])

        return np.squeeze(np.concatenate(outputs, axis=1))


