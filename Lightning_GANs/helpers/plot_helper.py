import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple, List


def get_subplot_adjustment(number_of_figures: int) -> Tuple[int, int]:
    # return number of rows and cols the closest to square value
    cols = round(math.sqrt(number_of_figures))
    rows = cols
    while rows * cols < number_of_figures:
        cols += 1
    return rows, cols


def plot_training_loss(minibatch_losses, num_epochs: int, averaging_iterations: int = 100, custom_label: str = '', set_y_axis_limit: bool = True, manual_y_axis_limit: List = None) -> None:

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    # USTAWIENIE LIMITU NA OSI y
    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    if(set_y_axis_limit):
        ax1.set_ylim([
            0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    if manual_y_axis_limit is not None:
        ax1.set_ylim(manual_y_axis_limit)

    # DORYSOWANIE ŚREDNIEJ KROCZĄCEJ https://doraprojects.net/questions/13728392/moving-average-or-running-mean
    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    # TWORZENIE OSI Z EPOKAMI
    ax2 = ax1.twiny()  # TWORZENIE DRUGIEJ OSI DLA TEGO SAMEGO y
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::num_epochs])
    ax2.set_xticklabels(newlabel[::num_epochs])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    # ###################

    plt.tight_layout()
