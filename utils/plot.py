import numpy as np
import time
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from os.path import join as pjoin
from matplotlib import pyplot as plt
from matplotlib import cm
from utils.file import makedir, load_from_file

def show_slices(array, filename=None, every=5, cols=10, figsize=(24, 12)):
    """Plot z-axis slices of the specified 3D array.

    Args:
        array (numpy.ndarray): Array to plot.
        filename (str): Path to save image. If None, no image is saved.
        every (int): To print all slices set this value to 1.
        cols (int): Number of columns in the figure.
        figsize (tuple): Figure size.
    """
    
    n = int(np.ceil(len(array)/every))
    rows = int(np.ceil(n/cols))

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i, idx in enumerate(range(0, len(array), every)):
        r = i // cols
        c = i % cols
        
        if rows == 1:
            axes = ax[c]
        elif cols == 1:
            axes = ax[r]
        else:
            axes = ax[r, c]
        
        axes.set_title('slice %d' % idx)
        axes.imshow(array[idx], cmap='gray')
        axes.axis('off')
    
    if filename:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close('all')
    
def plot_results(layers, clusters_vec, error_vec, network_err, save_path,
                 layers_k_errors_mat=None, layers_k_errors_mat2=None, additional_name=None):
    """
    plot the experiment results.
    graph 1 - number of codewords (clusters) vs. layer ind
    graph 2 - error vs. layer ind

    Args:
        layers: (np array, str) contains count_layers name arrange by the order of appearance in the net
        clusters_vec: (int) the number of optimal codewords in each layer in 'count_layers'.
        error_vec: (float) the error rate associated with the classifier of the optimal codewords in each
                        layer in 'count_layers'
        network_err: (float) the network error.
        save_path: (str) a directory path for saving results
    """

    # Graph 1
    t = time.strftime('%Y%m%d_%H%M%S')
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(layers, clusters_vec)
    ax[0].set_ylim(0, np.max(clusters_vec))
    ax[0].set_xlabel('layer\'s index')
    ax[0].set_ylabel('number of analysis codewords')
    ax[1].plot(layers, error_vec, layers, network_err)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('layer\'s index')
    ax[1].set_ylabel('error rate')
    title_name = 'Optimal K across count_layers '+additional_name
    fig.suptitle(title_name)

    plt.tight_layout()
    if additional_name is not None:
        saving_name = title_name+'_'+t+'.png'
    else:
        saving_name = 'Optimal K across layers_'+t+'.png'

    saving_subplots_path = pjoin(save_path, saving_name)
    makedir(save_path)
    plt.savefig(saving_subplots_path)
    plt.close(fig)

    if layers_k_errors_mat is not None:
        plt.rcParams.update({'xtick.labelsize': 22.0, 'ytick.labelsize': 22.0})
        fig = plt.figure()
        ax = fig.add_subplot(111)
        k_range = layers_k_errors_mat[0, 1:]
        layers_range = layers_k_errors_mat[1:, 0]
        n_layers = len(layers_range)

        colorsList = []
        for key, value in mcolors.TABLEAU_COLORS.items():
            colorsList.append(value)

        # shapes = ['o-', 'v-', 's-', 'p-', '+-', '^-', '1-', '*-', 'h-', '.-']
        for i in range(n_layers):
            ax.semilogx(k_range, layers_k_errors_mat[i+1, 1:], 'o-', label='%d' % layers_range[i],
                        color=colorsList[i], markersize=12, linewidth=3.0)

        if layers_k_errors_mat2 is not None:
            for i in range(len(layers_range)):
                ax.semilogx(k_range, layers_k_errors_mat2[i+1, 1:], linestyle='--', marker='o',
                            color=colorsList[i], markersize=12, linewidth=3.0)
        # Add first legend:  only labeled data is included
        leg1 = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.1), prop={'size': 20})
        # Add second legend for the maxes and mins.
        # leg1 will be removed from figure
        legend_elements = [Line2D([0], [0], color='black', lw=3, label='Discriminative'),
                           Line2D([0], [0], linestyle='--', color='black', lw=3, label='Generative')]
        leg2 = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.89, 1.0), prop={'size': 22})
        # Manually add the first legend back
        ax.add_artist(leg1)

        # plt.legend(loc='upper right', bbox_to_anchor=(1.09, 1.01), prop={'size': 18})
        plt.xlabel('Number of analysis words', fontsize=32)
        plt.ylabel('Error rate', fontsize=32)
        plt.xticks(k_range, k_range)
        # plt.axes.labelsize('large')
        plt.axis('tight')
        # plt.title('Error rate vs. number of Gaussians', fontsize=15)
        plt.tight_layout()
        if additional_name is not None:
            saving_name = 'errors_clusters_graph_' + additional_name + '_' + t + '.png'
        else:
            saving_name = 'errors_clusters_graph_' + t + '.png'

        saving_plot_path = pjoin(save_path, saving_name)
        plt.savefig(saving_plot_path)
        plt.close()


def plot_similarity_mat(data, labels, title, color='Greens', saving_path=None):

    n_rows = len(labels)
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    ax = fig.add_subplot(111)

    if n_rows <= 20:
        fontsize = 8
    elif n_rows <= 50:
        fontsize = 6
    else:
        fontsize = 4
    # cmap = cm.get_cmap(color)
    ax.matshow(data, interpolation='nearest')
    ax.tick_params(axis="x", labelbottom=True, labeltop=False, labelsize=26)
    ax.tick_params(axis="y", labelsize=26)
    # plt.suptitle(title, verticalalignment='bottom')
    # plt.xticks(range(n_rows), labels, fontsize=14)
    # plt.yticks(range(n_rows), labels, fontsize=14)
    # fig.colorbar(cax)
    plt.tight_layout()
    saving_plot_path = pjoin(saving_path, title+'.png')
    plt.savefig(saving_plot_path)
    plt.close()




