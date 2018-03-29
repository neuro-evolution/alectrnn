"""
Various functions for plotting and analyzing the output of the neural networks

(requires ffmpeg for animations)
"""

from alectrnn import handlers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


def ecdf(data):
    """
    Generates an empirical cumulative distribution function from data
    :param data: a 1D array of data
    :return: sorted data, cdf
    """
    data = np.sort(data)
    size = float(len(data))
    all_unique = not(any(data[:-1] == data[1:]))
    if all_unique:
        cdf = np.array([i / size for i in range(0,len(data))])
    else:
        cdf = np.searchsorted(data, data,side='left')/size
        unique_data, unique_indices = np.unique(data, return_index=True)
        data=unique_data
        cdf = cdf[unique_indices]

    return data, cdf


def eccdf(data):
    """
    Generates an empirical complementary cumulative distribution function
    :param data: a 1D array of data
    :return: sorted data, ccdf
    """
    sorted_data, cdf = ecdf(data)
    return sorted_data, 1. - cdf


def plot_ccdf(data, xlabel='', x_log=False, y_log=False, savefile=True,
              prefix='test', marker='o', linestyle='-', color='b',
              **kwargs):
    """
    Plots the CCDF of a given 1D array of data
    :param data: 1D array of data
    :param xlabel: label for x-axis
    :param x_log: True/false log x-axes
    :param y_log: True/false log y-axes
    :param savefile: True/false (if false, image is displayed)
    :param prefix: prefix for savefile name
    :param marker: marker argument for plt
    :param linestyle: marker argument for plt
    :param color: marker argument for plt
    :param kwargs: Any other plt.plot key word arguments
    :return: None
    """
    x, y = eccdf(data)
    plt.clf()
    plt.plot(x, y, marker=marker, linestyle=linestyle, color=color, **kwargs)
    if x_log: plt.xscale('log')
    if y_log: plt.yscale('log')
    plt.ylabel('CCDF')
    plt.xlabel(xlabel)
    if savefile:
        plt.savefig(prefix + '.png', dpi=300)
        plt.clf()
        plt.close()


def generate_color(index, loop_size=12, colormap=cm.Set1):
    """
    """

    return colormap((index%12 + index/12)*256 / loop_size)


def animate_screen(screen_history, prefix="test"):
    """
    Animates the ALE screen. Requires ffmpeg.
    :param screen_history: A HxWx3 numpy array (float or unint)
    :param prefix: prefix for output file name
    :return: None
    """
    fig = plt.figure()
    ims = []
    for frame in screen_history:
        im = plt.imshow(frame, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=17, blit=True,
                                    repeat_delay=1000)
    ani.save(prefix + '_screen_animation.mp4')


def animate_input(first_layer_state, input_shape, prefix="test"):
    """
    Animates the input frame for the neural network. Requires ffmpeg.
    :param first_layer_state: numpy array of states for the input layer
    :param input_shape: shape of layer (currently only supports 1xHxW)
    :param prefix: prefix for output file name
    :return: None
    """
    # TODO: make another anim for color/lumin after color is implemented (may need reshape)
    if first_layer_state.shape[1] != 1:
        raise NotImplementedError("Currently only makes animations for "
                                  "greyscale")

    fig = plt.figure()
    ims = []
    for state in first_layer_state:
        img = state[0].reshape(input_shape)
        im = plt.imshow(img, cmap=plt.get_cmap('Greys'), animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=17, blit=True,
                                    repeat_delay=1000)
    ani.save(prefix + '_input_animation.mp4')


def plot_internal_state(layer_state, index, neuron_ids, prefix="test"):
    """
    Makes a time series plot of all the neurons in a given layer
    :param layer_state: a Tx(L) matrix, where L is the shape of the layer
    :param index: the layer index (for labelling)
    :param neuron_ids: the indices location of neurons in the flattened state
    :param prefix: prefix for output file name
    :return: None
    """
    plt.clf()
    for neuron_state in layer_state.reshape(-1, layer_state.shape[0])[neuron_ids]:
        plt.plot(neuron_state, alpha=0.3, color="gray")

    plt.savefig(prefix + "_layer" + str(index) + "_states.pdf")
    plt.clf()
    plt.close()


def plot_internal_state_distribution(layer_state, index, prefix="test"):
    """
    Makes a time series plot of the max/min/median/90%/10% quartile ranges
    :param layer_state: a Tx(L) matrix, where L is the shape of the layer
    :param index: the layer index (for labelling)
    :param prefix: prefix for output file name
    :return: None
    """
    plt.clf()

    mins = []
    maxes = []
    tops = []
    bots = []
    centers = []
    for state in layer_state:
        values = state.flatten()

        # Max, min
        mins.append(min(values))
        maxes.append(max(values))

        # Percentile 90, 10, 50
        tops.append(np.percentile(values, 90))
        bots.append(np.percentile(values, 10))
        centers.append(np.percentile(values, 50))

    plt.plot(centers, color="black", marker="None", lw=2)
    plt.fill_between(range(len(tops)), tops, bots, color="red", alpha=0.3)
    plt.plot(mins, ls="--", marker="None", color="black")
    plt.plot(maxes, ls="--", marker="None", color="black")

    plt.savefig(prefix + "_layer" + str(index) + "_state_dist.pdf")
    plt.clf()
    plt.close()


def plot_output(last_layer_state, prefix="test"):
    """
    Time series plot of the trajectory of output states.
    :param last_layer_state: matrix of output layer
    :param prefix: prefix for output file name
    :return: None
    """
    plt.clf()
    for i, output in enumerate(last_layer_state.T):
        plt.plot(output, label="output: " + str(i))

    plt.legend()
    plt.savefig(prefix + "_output.pdf")
    plt.clf()
    plt.close()


def plot_parameter_distributions(parameters, parameter_layout, prefix="test",
                                 x_log=False, y_log=False, marker='o',
                                 linestyle='-', color='b', **kwargs):
    """
    Makes plots for the distributions of the parameters from the model.
    :param parameters: the parameter array (size N).
    :param parameter_layout: the int code for what type the parameters
        are (size N).
    :param prefix: prefix for output file name
    :param x_log: True/False log x-axes
    :param y_log: True/False log y-axes
    :param marker: plt.plot marker style (see matplotlib docs for options)
    :param linestyle: plt.plot linestyle (see matplotlib docs for options)
    :param color: plt.plot color (see matplotlib docs for options)
    :param kwargs: Any other key word arguments for plt.plot
    :return: None
    """

    for par_type in handlers.PARAMETER_TYPE:
        type_indices = np.where(parameter_layout == par_type.value)[0]
        if len(type_indices != 0):
            type_parameters = parameters[type_indices]
            type_prefix = prefix + "_type-" + par_type.name
            plot_ccdf(type_parameters, xlabel=par_type.name, prefix=type_prefix,
                      x_log=x_log, y_log=y_log, marker=marker,
                      linestyle=linestyle, color=color, **kwargs)


if __name__ == "__main__":
    pass
