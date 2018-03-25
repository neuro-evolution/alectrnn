"""
Various functions for plotting and analyzing the output of the neural networks

(requires ffmpeg for animations)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_screen():
    pass


def animate_input(first_layer_state, input_shape, prefix="test"):
    # TODO: make another anim for color/lumin after color is implemented (may need reshape)
    if input_shape[0] != 1:
        raise NotImplementedError("Currently only makes animations for "
                                  "greyscale")

    fig = plt.figure()
    ims = []
    for state in first_layer_state:
        img = state[0].reshape(input_shape)
        im = plt.imshow(img, cmap=plt.get_cmap('Greys'), animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat_delay=1000)
    ani.save(prefix + '_input_animation.mp4')


def plot_internal_states(layer_state, index, prefix="test"):

    plt.clf()

    for neuron_states in layer_state.T:
        plt.plot(neuron_states, alpha=0.3)

    plt.savefig(prefix + "_layer" + str(index) + "_states.pdf")
    plt.clf()
    plt.close()


def plot_internal_state_distribution(layer_state, index, prefix="test"):

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

    plt.clf()
    for i, output in enumerate(last_layer_state.T):
        # print(output.shape)
        # print(output)
        plt.plot(output, label="output: " + str(i))

    plt.savefig(prefix + "_output.pdf")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    pass
