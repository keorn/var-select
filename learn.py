from __future__ import print_function   # Backward compatibility
import h5py
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import itertools
import numpy as np
import lasagne
import time
#import seaborn as sns

NUM_EPOCHS = 20         # Maximum number of epochs to train
BATCH_SIZE = 1          # Batches dont work now
NUM_HIDDEN_UNITS = 10   # Number of units in the hidden layers
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def load_data():
    """ Make a dictionary with all stuff related to the data """
    data = h5py.File("dataset.jld", "r+")
    X_train = np.transpose(data["X_train"])
    y_train = np.transpose(data["y_train"])
    X_valid = np.transpose(data["X_valid"])
    y_valid = np.transpose(data["y_valid"])
    X_test  = np.transpose(data["X_test"])
    y_test  = np.transpose(data["y_test"])
    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=theano.shared(lasagne.utils.floatX(y_train)),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=theano.shared(lasagne.utils.floatX(y_valid)),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=theano.shared(lasagne.utils.floatX(y_test)),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=1,
    )


def build_model(input_dim, output_dim,
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
    """ Model with two hidden layers, rectified units, dropout = 0.5 """
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden1_dropout = lasagne.layers.DropoutLayer(
        l_hidden1,
        p=0.5,
    )
    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(
        l_hidden2,
        p=0.5,
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    return l_out


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    """ Create all functions needed for training """
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.dvector('y')
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.mse)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval  = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    pred = output_layer.get_output(X_batch, deterministic=True)
    accuracy = T.sqrt(lasagne.objectives.mse(y_batch, pred))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
    )


def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    """ Create a generator used for training """

    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }


def test(iter_funcs, dataset):
    """ Calculate accuracy on the test data """
    test_losses = []
    test_accuracies = []
    for j in range(dataset['num_examples_test']):
        test_loss, test_accuracy = iter_funcs['test'](j)
        test_accuracies.append(test_accuracy)

    avg_test_accuracy = np.mean(test_accuracies)

    return dict(
        test_accuracy=avg_test_accuracy,
    )


def plot_rand(dataset, output_layer):
    """Plot a random test instance"""
    j = np.random.randint(dataset['X_test'].get_value().shape[0])
    X_test = dataset['X_test'].get_value()[j]
    X = X_test[:100]
    Y = X_test[100:]
    true_R2 = 1 - dataset['y_test'].get_value()[j]
    pred = 1 - output_layer.get_output(X_test).eval()[0, 0]
    print('True test R2 = %01.3f' % true_R2)
    print('Pred test R2 = %01.3f' % pred)
    plt.scatter(X, Y)
    plt.show()


def pred_scatter(dataset, output_layer, n_plot=10):
    """Scatter plot of predicted values and true values"""
    # FIXME - I'm yet to use theano so this is definitely an inefficient way to do things!
    n = dataset['X_test'].get_value().shape[0]
    true = []
    pred = []
    for dummy in range(n_plot):
        print('%d of %d' % (dummy + 1, n_plot))
        j = np.random.randint(n)
        true.append(dataset['y_test'].get_value()[j])
        pred.append(output_layer.get_output(dataset['X_test'].get_value()[j]).eval()[0, 0])
    plt.scatter(true, pred)
    plt.show()


def main(num_epochs=NUM_EPOCHS):
    print("Loading data...")
    dataset = load_data()

    print("Building model and compiling functions...")
    output_layer = build_model(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
    )
    iter_funcs = create_iter_functions(dataset, output_layer)

    print("Starting training...")
    now = time.time()
    previous_valid_loss = np.inf
    try:
        for epoch in train(iter_funcs, dataset):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f}".format(epoch['valid_accuracy']))

            if epoch['number'] >= num_epochs:
                break

            if previous_valid_loss <= epoch['valid_loss']:
                break
            previous_valid_loss = epoch['valid_loss']

    except KeyboardInterrupt:
        pass

    print("Testing...")
    test_result = test(iter_funcs, dataset)
    print("Test accuracy:\t\t\t{:.6f}".format(test_result['test_accuracy']))

    return output_layer


if __name__ == '__main__':
    main()
