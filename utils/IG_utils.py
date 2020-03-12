import matplotlib.patches as patches
import numpy as np
import os
import copy
from scipy.io import savemat
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from os.path import join as pjoin
from layers import get_layer_output
from utils.file import load_from_file, makedir
from utils.gmm_utils import get_most_likely_clusters, calc_clusters_density_hist, spatial_pairwise_hist


def gather_coOccurrance_mat(gmm_model, exp_path, RF, layers_to_inference, data,
                            num_of_iterations=10, saving_flag=True, return_flag=True):
    # TODO the data needs to be with mean subtraction
    n_data = data.shape[0]

    interval = np.floor_divide(n_data, num_of_iterations)
    appearance_dict = {}

    for i in range(num_of_iterations):
        print(f'begin iterate {i+1}/{num_of_iterations}')
        if i + 1 == num_of_iterations:
            preds = gmm_model.predict(data[i * interval:], batch_size=10)
        else:
            preds = gmm_model.predict(data[i * interval:(i + 1) * interval], batch_size=10)

        print('finish predicting')

        explained_layer_name = layers_to_inference[-1]

        for explaining_layer_name in reversed(layers_to_inference[:-1]):

            explained_gmm_layer_name = gmm_model.get_correspond_gmm_layer(explained_layer_name)
            explaining_gmm_layer_name = gmm_model.get_correspond_gmm_layer(explaining_layer_name)
            explained_layer = gmm_model.keras_model.get_layer(explained_layer_name)

            if type(explained_layer).__name__ == 'Dense':
                explained_k_mat = get_most_likely_clusters(preds['GMM'][explained_gmm_layer_name])
                explaining_k_mat = get_most_likely_clusters(preds['GMM'][explaining_gmm_layer_name])

                n_explained_clusters = preds['GMM'][explained_gmm_layer_name].shape[-1]
                n_explaining_clusters = preds['GMM'][explaining_gmm_layer_name].shape[-1]

                tot_clusters_hist = calc_clusters_density_hist(n_explained_clusters=n_explained_clusters,
                                                               n_explaining_clusters=n_explaining_clusters,
                                                               explained_clusters=explained_k_mat,
                                                               explaining_clusters=explaining_k_mat)

            else:
                tot_clusters_hist = spatial_pairwise_hist(rf=RF,
                                                          explained_layer_name=explained_layer_name,
                                                          explaining_layer_name=explaining_layer_name,
                                                          explained_gmm_preds=preds['GMM'][explained_gmm_layer_name],
                                                          explaining_gmm_preds=preds['GMM'][explaining_gmm_layer_name])

            if explained_layer_name + "-" + explaining_layer_name in appearance_dict:
                appearance_dict[explained_layer_name + "-" + explaining_layer_name] += tot_clusters_hist
            else:
                appearance_dict[explained_layer_name + "-" + explaining_layer_name] = tot_clusters_hist

            # Update dependencies
            explained_layer_name = explaining_layer_name

    # Saving each mat separately
    if saving_flag:
        results_dir = pjoin(exp_path, 'results')
        makedir(results_dir)
        for key in appearance_dict:
            savemat(pjoin(results_dir, key + '.mat'),
                    mdict={key: appearance_dict[key]})
    if return_flag:
        return appearance_dict


def create_connections_heat_maps(gmm_model, exp_path,  RF, data, connections_dict, saving_dir):
    """ This script creates the heat map between two clusters connected in the inference tree"""
    # TODO the data needs to be with mean subtraction
    makedir(saving_dir)

    clusters_rep_path = pjoin(exp_path, 'clusters_representatives.json')

    if os.path.isfile(clusters_rep_path) and os.access(clusters_rep_path, os.R_OK):
        # checks if file exists
        print("File exists and is readable")
        print("Load results...")
        clusters_representatives = load_from_file(exp_path, ['clusters_representatives'])[0]

    else:
        raise ValueError("Either file is missing or is not readable")

    for explained_key in connections_dict:
        if 'class' in explained_key or 'fc' in explained_key:
            continue

        explained_layer = explained_key.split('_')[0]
        for i in range(len(explained_key.split('_')) - 2):
            explained_layer = explained_layer + '_' + explained_key.split('_')[i+1]

        explained_cluster = explained_key.split('_')[-1]
        explained_gmm = gmm_model.get_correspond_gmm_layer(explained_layer)

        c_reps = clusters_representatives[explained_gmm][explained_cluster]
        c_imgs_indice = np.asarray(c_reps['image'])
        c_imgs = data[c_imgs_indice]

        if len(connections_dict[explained_key]) != 0:
            explaining_key = next(iter(connections_dict[explained_key]))
            explaining_layer = explaining_key.split('_')[0] + '_' + explaining_key.split('_')[1]

            explaining_gmm = gmm_model.get_correspond_gmm_layer(explaining_layer)

            imgs_pred = gmm_model.predict(c_imgs)

            explaining_pred = imgs_pred['GMM'][explaining_gmm]

            for explaining_key in connections_dict[explained_key]:
                explaining_layer = explaining_key.split('_')[0] + '_' + explaining_key.split('_')[1]
                explaining_cluster = int(explaining_key.split('_')[2]) - 1

                saving_plot_path = pjoin(saving_dir, explained_key + '_' + explaining_key + '.png')

                if os.path.isfile(saving_plot_path) and os.access(saving_plot_path, os.R_OK):
                    # checks if heat map file exists
                    continue

                _, _, origin_size = RF.target_neuron_rf(explained_layer, [0, 0],
                                                        rf_layer_name=explaining_layer, return_origin_size=True)
                heat_map = np.zeros((origin_size[0], origin_size[1]))
                for i in range(len(c_imgs_indice)):
                    h = c_reps['spatial_location']['row'][i]
                    w = c_reps['spatial_location']['col'][i]

                    size, center, UL_pos, origin_center = RF.target_neuron_rf(explained_layer, [h, w],
                                                                              rf_layer_name=explaining_layer,
                                                                              return_upper_left_pos=True,
                                                                              return_origin_center=True)

                    row_offset = abs(min(0, origin_center[0] - np.floor_divide(origin_size[0], 2)))
                    col_offset = abs(min(0, origin_center[1] - np.floor_divide(origin_size[1], 2)))
                    upper_left_row = UL_pos[0]
                    upper_left_col = UL_pos[1]

                    rf_preds = explaining_pred[i, upper_left_row:upper_left_row + size[0],
                                               upper_left_col:upper_left_col + size[1], explaining_cluster]

                    pad_rf = np.zeros((origin_size[0], origin_size[1]))
                    pad_rf[row_offset:rf_preds.shape[0] + row_offset, col_offset:rf_preds.shape[1] + col_offset] = rf_preds
                    heat_map += pad_rf

                heat_map = heat_map / len(c_imgs_indice)
                fig = plt.figure()
                label = str(connections_dict[explained_key][explaining_key]['CN']) + ', ' +\
                        str(connections_dict[explained_key][explaining_key]['LR'])
                ax = fig.add_subplot(111)
                ax.matshow(heat_map, interpolation='hermite')
                ax.axis('off')
                ax.set_title(label=label, fontsize=70)
                plt.tight_layout()

                plt.savefig(saving_plot_path)
                plt.close(fig)


def get_IG_connections_dict_LR(class_index, global_appearance_hist_dict, local_appearance_hist_dict,
                               layers_to_inference, n_nodes=2):
    """ Create nodes and edges for the inference tree according to class inference or image & class inference.

    Args:
        class_index (int) - the class index to be inference.
        global_appearance_hist_dict (dict) - a dictionary of co-occurrence matrices. Each key is a string from the format
            "explained_layer_name+'-'+explaining_layer_name" contained their co-occurrence matrix. This dictionary is
            built by gather_pair-wise_appearance.py script
        clusters_stats (dict) - dictionary of the clusters' statistics as the output of gatherCluster&classStats_v1.py
        layers_to_inference (vec) - the layers indices to inference, ordered from first to last as it appears
            in the network.
        gmm_model - gmm_classification_model obj.
        n_nodes(int) - the number of nodes (i.e. clusters) to present in each layer.
        image_inference (opt, boolean) - whether to pick the clusters to represent a layer according to image & class
            inference or class inference only. True- both, False- class only.
        image (opt, np.mat) - if image_inference field is true, image is the image pixels value to be inference.

    Output:
        connections_dict - dictionary contained the nodes & edged of the graph. Each key is the node i.d from the format
            'layer_%layerNumber_%clusterNumber', contains a dictionary of the node's edges as values, contains it's
             connections strength as the second dict' values.
             :param class_index:
             :param local_appearance_hist_dict:


    """
    connections_dict = {}

    explained_layer_name = layers_to_inference[-1]
    rep_clusters = [class_index]

    for explaining_layer_name in reversed(layers_to_inference[:-1]):
        pairwise_name = explained_layer_name+'-'+explaining_layer_name

        tot_clusters_hist = global_appearance_hist_dict[pairwise_name]
        C_t_s = local_appearance_hist_dict[pairwise_name]
        C_s = np.sum(C_t_s, axis=1)
        norm_C_t_s = np.transpose(np.transpose(C_t_s) / (C_s))

        # P(h'|h)
        sum_cols = np.sum(tot_clusters_hist, axis=1)
        clusters_prob_mat = np.transpose(np.transpose(tot_clusters_hist) / sum_cols)

        # P(h'=k)
        sum_rows = np.sum(tot_clusters_hist, axis=0)
        explaining_prior = sum_rows / np.sum(sum_rows)

        log_ratio = np.log2(clusters_prob_mat / explaining_prior)
        scores_mat = C_t_s*log_ratio
        scores_mat = np.nan_to_num(scores_mat)

        new_rep_clusters_prob = scores_mat[rep_clusters,:]
        clusters_LR_score = np.sum(new_rep_clusters_prob, axis=0)

        subset_S = np.where(clusters_LR_score > 0)[0]
        subset_S_power = clusters_LR_score[subset_S]

        new_rep_clusters = subset_S[np.argsort(-subset_S_power)]

        if len(new_rep_clusters) > n_nodes:
            new_rep_clusters = new_rep_clusters[:n_nodes]

        for explained_cluster in rep_clusters:
            if 'class' in explained_layer_name:
                cluster_key = 'class'
            else:
                cluster_key = explained_layer_name + '_' + str(explained_cluster + 1)
            new_rep_clusters_key = {}
            for explaining_cluster in new_rep_clusters:
                new_rep_clusters_key.update(
                    {explaining_layer_name + '_' + str(explaining_cluster + 1):
                         {'LR': round(log_ratio[explained_cluster, explaining_cluster], 2),
                          'CN': round(norm_C_t_s[explained_cluster, explaining_cluster], 2)}})

            connections_dict[cluster_key] = copy.deepcopy(new_rep_clusters_key)

    # Update dependencies
        explained_layer_name = explaining_layer_name
        rep_clusters = new_rep_clusters

    # Set the first layer as nodes without connections
    for given_cluster in rep_clusters:
        cluster_key = explained_layer_name + '_' + str(given_cluster+1)
        connections_dict[cluster_key] = []

    return connections_dict


def create_clusters_for_imageGraph(vis_image, image_array, image_false_label, true_label, gmm_model, rf,
                                   connections_dict, clusters_dir, saving_dir, cluster_type, image_pos='r'):
    makedir(saving_dir)
    for layer_cluster_key in connections_dict:
        # fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.001})
        # fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.0})
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        if layer_cluster_key == 'class':
            layer_name = 'classification'
            fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.0})
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            k_name = 'cluster_' + str(image_false_label+1) + '.png'
            k_dir = pjoin(*[clusters_dir, layer_name, k_name])
            cluster_img = mpimg.imread(k_dir)

            ax[0].imshow(cluster_img)
            ax[1].imshow(vis_image)
            ax[1].set_title(true_label, fontdict={'fontsize': 20.0})

        else:
            layer_type, layer_num, k_num = layer_cluster_key.split('_')
            layer_name = layer_type + '_' + layer_num
            k_name = 'cluster_' + k_num + '_' + cluster_type[layer_name] + '.png'
            if image_pos == 'r':
                fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.001})
            elif image_pos == 't':
                fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.0})
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            k_dir = pjoin(*[clusters_dir, layer_name, k_name])
            cluster_img = mpimg.imread(k_dir)

            radius_size = 0.5

            if image_pos == 't':
                ax[1].imshow(cluster_img)
                ax[0].imshow(vis_image)
            elif image_pos == 'r':
                ax[0].imshow(cluster_img)
                ax[1].imshow(vis_image)

            gmm_explaining_layer_name = gmm_model.get_correspond_gmm_layer(layer_name)
            corr_activation_layer = gmm_model.gmm_activations_dict[gmm_explaining_layer_name]
            img_pred = get_layer_output(keras_model=gmm_model.keras_model, layer_name=corr_activation_layer,
                                        input_data=image_array)
            H_dim = img_pred.shape[1]
            W_dim = img_pred.shape[2]
            K_dim = img_pred.shape[3]

            img_pred = np.reshape(img_pred, (H_dim * W_dim, K_dim))
            img_k = np.argmax(img_pred, axis=1)
            img_values = np.max(img_pred, axis=1)

            location_inds = np.where(img_k == (int(k_num)-1))[0]
            rep_values = img_values[location_inds]
            sor_location_inds = location_inds[np.argsort(-rep_values)]

            n_samples = W_dim*H_dim
            n_samples_in_k = len(sor_location_inds)

            first_time_flag = False # for adding a rectangle around first point - this field should be true
            for loc_i in sor_location_inds:
                xPos = np.remainder(loc_i, W_dim)
                yPos = np.floor_divide(loc_i, W_dim)

                size, center, upper_left_pos, origin_center = rf.target_neuron_rf(layer_name, (yPos, xPos),
                                                                                  rf_layer_name='input_layer',
                                                                                  return_origin_center=True,
                                                                                  return_upper_left_pos=True)

                # The upper and left rectangle coordinates
                upper_left_row = upper_left_pos[0]
                upper_left_col = upper_left_pos[1]

                if first_time_flag:
                    rect = patches.Rectangle((upper_left_col, upper_left_row), size[1] - 1, size[0] - 1,
                                              linewidth=2, edgecolor='r', facecolor='none')
                    # ax[0].add_patch(rect)
                    ax[1].add_patch(rect)
                    first_time_flag = False

                dot = patches.Circle((origin_center[1], origin_center[0]), radius=radius_size,
                                      linewidth=1, edgecolor='r', facecolor='r')
                # ax[0].add_patch(dot)
                ax[1].add_patch(dot)

            # ax[0].set_title(f'{n_samples_in_k}/{n_samples}', size=12)
            ax[1].set_title(f'{n_samples_in_k}/{n_samples}', size=12)

        ax[0].axis('off')
        ax[1].axis('off')
        plt.savefig(pjoin(saving_dir, layer_name + '_' + k_name), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

