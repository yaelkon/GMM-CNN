import numpy as np
import copy
from layers import get_layer_output
from tensorflow import set_random_seed

np.random.seed(101)
set_random_seed(101)

def get_most_likely_clusters(gmm_preds):
    """Returns the most likely cluster each spatial example is associated with, and keep the spatial dimension
    Args:
        gmm_preds(np array) - size (B, H, W, K) with the probability of P(h=k|x) in each cell as returns from
                              gmm_model.predict function.
    Returns:
          clusters_mat(np array) - size (B, H, W) with the index of the most likely cluster in each cell
                                    (i.e. each spatial example).
        """
    if isinstance(gmm_preds, list):
        n_gmm_layers = len(gmm_preds)
    else:
        n_gmm_layers = 1

    clusters_layers_list = []
    for i in range(n_gmm_layers):

        if isinstance(gmm_preds, list):
            preds_tens = gmm_preds[i]
        else:
            preds_tens = gmm_preds

        tens_shape = np.shape(preds_tens)

        if len(tens_shape) == 2:
            arg_max_tens = np.argmax(preds_tens, axis=1)

        elif len(tens_shape) == 4:
            B_dim = tens_shape[0]
            H_dim = tens_shape[1]
            W_dim = tens_shape[2]
            n_samples = B_dim * H_dim * W_dim

            K_dim = tens_shape[3]
            re_tens = np.reshape(preds_tens, (n_samples, K_dim))
            re_argsmax_tens = np.argmax(re_tens, axis=1)
            arg_max_tens = np.reshape(re_argsmax_tens, (B_dim, H_dim, W_dim))

        clusters_layers_list.append(arg_max_tens)

    if n_gmm_layers == 1:
        return clusters_layers_list[0]
    else:
        return clusters_layers_list


def find_max(outputs_gmm_tens, outputs_llr_tens=None, n_max=10):
    """Finds the most likely n_max examples of each cluster
    Args:
        outputs_gmm_tens (list) - a 4D tensor size (n_examples, height, weight, n_gaussians), containing the p(h|x) of
         each examples. The output of 'GMM' field of gmm_model predict func.

        outputs_llr_tens (list) (optional) - a 4D tensor size (n_examples, height, weight, n_gaussians), containing the
         llr of each examples. For getting this output the field 'set_gmm_as_output' gmm_model has to be true before
         model creation. If this field is not None- among of all examples with p(h=h*|X)=1 they will be ordered by the largesr llr.

        n_max (int) - the number of examples with the highest llr to return for 'save_version_preds_list'.

    Returns:
        llr_max (dict) - 'values' (nd array) - the score of n_max examples for each gaussian.
                         'args' (list) - containing 2 arrays: 1)example index (of the image), 2)spatial location
        save_version_preds_list (list) - same as llr_max but smaller version with only n_max representatives to save.
    """
    if isinstance(outputs_gmm_tens, dict):
        n_gmm_layers = len(outputs_gmm_tens)
    else:
        n_gmm_layers = 1
    if outputs_llr_tens is not None:
        order_by_llr = True
    else:
        order_by_llr = False

    layers_preds_list = []
    save_version_preds_list = []
    for i, key in zip(range(n_gmm_layers), outputs_gmm_tens):
        if isinstance(outputs_gmm_tens, dict):
            preds_tens = outputs_gmm_tens[key]
            if order_by_llr:
                llr_tens = outputs_llr_tens[key]
        else:
            preds_tens = outputs_gmm_tens
            if order_by_llr:
                llr_tens = outputs_llr_tens

        tens_shape = np.shape(preds_tens)
        if len(tens_shape) == 4:
            B_dim = tens_shape[0]
            H_dim = tens_shape[1]
            W_dim = tens_shape[2]
            K_dim = tens_shape[3]
            n_samples = B_dim*H_dim*W_dim
            re_tens = np.reshape(preds_tens, (n_samples, K_dim))
            if order_by_llr:
                llr_re_tens = np.reshape(llr_tens, (n_samples, K_dim))
        elif len(tens_shape) == 2:
            K_dim = tens_shape[-1]
            re_tens = preds_tens
            if order_by_llr:
                llr_re_tens = llr_tens

        preds_k = np.argmax(re_tens, axis=1)
        clusters_dict = {}
        save_version_cluster_dict = {}

        for k in range(K_dim):
            k_indices = np.where(preds_k == k)[0]
            k_values = re_tens[k_indices, k]
            k_values = np.round(k_values, 3)

            argsort = np.argsort(k_values)
            k_indices = np.flip(k_indices[argsort])
            k_values = np.flip(k_values[argsort])
            if order_by_llr:
                llr_value = llr_re_tens[k_indices, k]
            threshold_ind = 0

            if k_values.size != 0:
                if order_by_llr and k_values[0] == 1:
                    threshold_ind = np.where(k_values == 1)[0][-1]
                #     re_ordered_indices = k_indices[:(threshold_ind + 1)]
                #     llr_value = llr_re_tens[re_ordered_indices, k]
                #     argsort = np.argsort(-llr_value)
                #     re_ordered_indices = re_ordered_indices[argsort]
                #     k_indices[:(threshold_ind + 1)] = re_ordered_indices

                if len(tens_shape) == 4:
                    image_inds = np.ceil((k_indices + 1) / (H_dim * W_dim)).astype('int32') - 1
                    # location_inds = np.remainder(k_indices, H_dim * W_dim)
                    # xPos = np.remainder(location_inds, W_dim)
                    # yPos = np.floor_divide(location_inds, W_dim)

                elif len(tens_shape) == 2:
                    image_inds = k_indices
                    # xPos = None
                    # yPos = None

                clusters_dict[k + 1] = {}
                clusters_dict[k + 1] = image_inds
                save_version_cluster_dict[k + 1] = {}
                if len(k_indices) > n_max:
                    temp_max = n_max
                    if threshold_ind > n_max:
                        temp_max = threshold_ind

                    save_version_cluster_dict[k + 1]['k_indices'] = k_indices[:temp_max]
                    save_version_cluster_dict[k + 1]['values'] = k_values[:temp_max]
                    if order_by_llr:
                        save_version_cluster_dict[k + 1]['llr'] = llr_value[:temp_max]
                else:
                    save_version_cluster_dict[k + 1]['k_indices'] = k_indices
                    save_version_cluster_dict[k + 1]['values'] = k_values
                    if order_by_llr:
                        save_version_cluster_dict[k + 1]['llr'] = llr_value
            else:
                clusters_dict[k + 1] = None
                save_version_cluster_dict[k + 1] = None

        layers_preds_list.append(clusters_dict)
        save_version_preds_list.append(save_version_cluster_dict)

    return layers_preds_list, save_version_preds_list

# option 1
# argsorted_tens = np.argsort(-re_tens, axis=0)
# k_max_args = argsorted_tens[:n_max,:]
# re_k_max_arg = np.reshape(k_max_args, n_max*K_dim)
# k_max_values = re_tens[re_k_max_arg, np.where(k_max_args == k_max_args)[-1]]
# k_max_values = np.reshape(k_max_values, (n_max, K_dim))
# # Calc the image index
#
# if len(tens_shape) == 4:
#     k_max_image_inds = np.ceil((k_max_args+1)/(H_dim*W_dim)).astype('int32') - 1
#     k_max_location_inds = np.remainder(k_max_args, H_dim*W_dim)
#     k_max_xPos = np.remainder(k_max_location_inds, W_dim)
#     k_max_yPos = np.floor_divide(k_max_location_inds, W_dim)
#
# elif len(tens_shape) == 2:
#     k_max_image_inds = k_max_args
#     k_max_xPos = None
#     k_max_yPos = None

# k_max_location_inds[k_max_location_inds == 0] = H_dim*W_dim

# llr_max = {}
# llr_max['values'] = np.transpose(k_max_values).tolist()
# llr_max['spatial_location'] = {'col': np.transpose(k_max_xPos).tolist()}
# llr_max['spatial_location'].update({'row': np.transpose(k_max_yPos).tolist()})
# llr_max['image'] = np.transpose(k_max_image_inds).tolist()
#
# llr_layers_list.append(llr_max)
# End option 1

def get_cluster_reps(clusters_dict, H_dim=None, W_dim=None, n_reps=10):
    dict = {}
    for cluster in clusters_dict:
        if clusters_dict[cluster] is not None:
            k_indices = clusters_dict[cluster]['k_indices']
            k_values = clusters_dict[cluster]['values']

            if 'llr' in clusters_dict[cluster]:
                k_llr = clusters_dict[cluster]['llr']

            argsort = np.argsort(k_values)
            k_indices = np.flip(k_indices[argsort])
            k_values = np.flip(k_values[argsort])
            try:
                k_llr

            except NameError:
                llr_exists = False
            else:
                llr_exists = True
            if llr_exists:
                k_llr = np.flip(k_llr[argsort])

            # if k_values.size != 0:
            if llr_exists and k_values[0] == 1:
                threshold_ind = np.where(k_values == 1)[0][-1]
                one_value_indices = k_indices[:(threshold_ind + 1)]
                llr_value = k_llr[:(threshold_ind + 1)]
                # llr_value = llr_re_tens[one_value_indices, k]
                argsort = np.argsort(-llr_value)
                one_value_indices = one_value_indices[argsort]
                k_indices[:(threshold_ind + 1)] = one_value_indices
                k_llr[:(threshold_ind + 1)] = llr_value[argsort]

            new_n_rep = n_reps
            if len(k_indices) < n_reps:
                new_n_rep = len(k_indices)
            k_indices = k_indices[:new_n_rep]
            k_values = k_values[:new_n_rep]
            k_llr = k_llr[:new_n_rep]
            if H_dim is not None and W_dim is not None:
                image_inds = np.ceil((k_indices + 1) / (H_dim * W_dim)).astype('int32') - 1
                location_inds = np.remainder(k_indices, H_dim * W_dim)
                xPos = np.remainder(location_inds, W_dim)
                yPos = np.floor_divide(location_inds, W_dim)

            else:
                image_inds = k_indices
                xPos = None
                yPos = None

            dict[cluster] = {}
            # dict[cluster]['values'] = np.transpose(k_values).tolist()
            dict[cluster]['spatial_location'] = {'col': np.transpose(xPos).tolist()}
            dict[cluster]['spatial_location'].update({'row': np.transpose(yPos).tolist()})
            dict[cluster]['image'] = np.transpose(image_inds).tolist()
            # dict[cluster]['llr'] = np.transpose(k_llr).tolist()

    return dict


def calc_bayes_depended_prob(TLLs):
    """Computes the probabiliity P(h=k|x), where TLL is the output of gmm layer is given by log(P(x,h=k))"""
    if isinstance(TLLs, list):
        num_TLLs = len(TLLs)
    else:
        num_TLLs = 1

    depended_probs_list = []

    for i in range(num_TLLs):
        if isinstance(TLLs, list):
            TLL = TLLs[i]
        else:
            TLL = TLLs

        shape = np.shape(TLL)
        K_dim = shape[-1]
        max_TLL = np.amax(TLL, axis=-1)

        if len(shape) == 2:
            max_TLL = np.repeat(max_TLL[:, np.newaxis], K_dim, axis=-1)
        elif len(shape) == 4:
            max_TLL = np.repeat(max_TLL[:,:,:, np.newaxis], K_dim, axis=-1)

        ETLL = np.exp(TLL - max_TLL)
        SETLL = np.sum(ETLL, -1)

        if len(shape) == 2:
            rep_SETLL = np.repeat(SETLL[:, np.newaxis], K_dim, axis=-1)
        elif len(shape) == 4:
            rep_SETLL = np.repeat(SETLL[:,:,:, np.newaxis], K_dim, axis=-1)

        depended_prob = ETLL / rep_SETLL
        depended_probs_list.append(depended_prob)

    return depended_probs_list

def set_gmm_weights(keras_model, layer_gmm_params_dict, set_std=True, initialization=True):
    """for each gmm layer set the mean and std weights to be the mean and std of conv layer it modeled.
    Args:
        keras_model - keras model
        layer_gmm_params_dict - a dict containing the gmm count_layers names as keys. each key is a dict containing the keys
        'mu' and 'std' as it returns from 'calc_layer_mean_and_std()' func below. the values must be as the size of the
         channels of the conv layer (1, channels).
    Returns:
        keras_model - the same model after the weights has been initiated.
        """
    # for layer in keras_model.count_layers:
    for gmm_layer_name in layer_gmm_params_dict:
        layer = keras_model.get_layer(gmm_layer_name)
        weights = layer.get_weights()

        if type(layer).__name__ == 'GMM':
            if initialization:
                if layer.name == 'gmm_classification':
                    classifier_layer_name = 'classifier_' + layer.name
                    classifier_layer = keras_model.get_layer(classifier_layer_name)
                    classifier_weights = classifier_layer.get_weights()

                    weights[0] = np.ones_like(weights[0])*layer_gmm_params_dict[layer.name]['mu'][0]
                    classifier_weights[0] = np.ones_like(classifier_weights[0])*\
                                            layer_gmm_params_dict[layer.name]['mu'][0]
                    classifier_layer.set_weights(classifier_weights)
                else:
                    if len(layer_gmm_params_dict[layer.name]['mu'][0].shape) == 1:
                        weights[0] = np.ones_like(weights[0]) * \
                                     np.random.multivariate_normal(mean=layer_gmm_params_dict[layer.name]['mu'][0],
                                                                   cov=np.diag(pow(layer_gmm_params_dict[layer.name]['std'][0], 2)),
                                                                   size=weights[0].shape[0])
                    else:
                        weights[0] = np.ones_like(weights[0]) * layer_gmm_params_dict[layer.name]['mu'][0]

                if set_std:
                    # weights[1] = np.ones_like(weights[1]) * layer_gmm_params_dict[layer.name]['std'][0]
                    weights[1] = np.ones_like(weights[1]) * np.sqrt(layer_gmm_params_dict[layer.name]['std'][0])
            else:
                weights[0] = np.ones_like(weights[0]) * layer_gmm_params_dict[layer.name]['mu']
                if set_std:
                    weights[2] = np.ones_like(weights[2]) * layer_gmm_params_dict[layer.name]['alpha']
                else:
                    weights[1] = np.ones_like(weights[1]) * layer_gmm_params_dict[layer.name]['alpha']

                if set_std:
                    weights[1] = np.ones_like(weights[1]) * layer_gmm_params_dict[layer.name]['std']

        else:
            weights[0] = np.ones_like(weights[0]) * layer_gmm_params_dict[layer.name]['w']
            weights[1] = np.ones_like(weights[1]) * layer_gmm_params_dict[layer.name]['b']

        layer.set_weights(weights)

    return keras_model

def get_gmm_weights_dict(gmm_model):
    """returns the weights for all gmm count_layers"""
    gmm_weights_dict = {}
    for key in gmm_model.gmm_dict:
        gmm_layer_name = gmm_model.gmm_dict[key]
        gmm_weights_dict[gmm_layer_name] = {'mu': {}, 'std': {}, 'alpha': {}}

        layer = gmm_model.keras_model.get_layer(gmm_layer_name)
        weights = layer.get_weights()

        gmm_weights_dict[gmm_layer_name]['mu'] = weights[0]
        # if gmm_model.train_std:
        gmm_weights_dict[gmm_layer_name]['std'] = weights[1]
        gmm_weights_dict[gmm_layer_name]['alpha'] = weights[2]
        # else:
        #     gmm_weights_dict[gmm_layer_name]['std'] = weights[2]
        #     gmm_weights_dict[gmm_layer_name]['alpha'] = weights[1]

        if gmm_model.training_method == 'discriminative' and gmm_layer_name in gmm_model.classifiers_dict:
            classifier_layer_name = gmm_model.classifiers_dict[gmm_layer_name]
            gmm_weights_dict[classifier_layer_name] = {'w': {}, 'b': {}}

            layer = gmm_model.keras_model.get_layer(classifier_layer_name)
            weights = layer.get_weights()

            gmm_weights_dict[classifier_layer_name]['w'] = weights[0]
            gmm_weights_dict[classifier_layer_name]['b'] = weights[1]

    return gmm_weights_dict

def calc_layer_mean_and_std(gmm_model, data, keras_model=None, mode=1):
    """For each conv layer calc the mean and std of the output data

    mode - 1 - returns the mean of the data.
           2 - returns K examples randomly picked as representatives. """

    if not keras_model:
        keras_model = gmm_model.keras_model

    layer_gmm_params = {}

    for layer_name, gmm_layer_name in gmm_model.gmm_dict.items():
        layer_gmm_params[gmm_layer_name] = {}

        if layer_name == 'classification':
            x = np.zeros((keras_model.get_layer(gmm_layer_name).K,
                         keras_model.get_layer(gmm_layer_name).K))
            np.fill_diagonal(x, 1)
            layer_gmm_params[gmm_layer_name] = {}
            layer_gmm_params[gmm_layer_name].update({'mu': [x]})
            layer_gmm_params[gmm_layer_name].update({'std': [0.1]})

        else:
            x = get_layer_output(keras_model, layer_name, data)
            K = keras_model.get_layer(gmm_layer_name).K
            if len(x.shape) == 4:
                B_dim = x.shape[0]
                H_dim = x.shape[1]
                W_dim = x.shape[2]
                D_dim = x.shape[3]
                n_samples = B_dim * H_dim * W_dim
                x = np.reshape(x, (n_samples, D_dim))
            elif len(x.shape) == 2:
                n_samples = x.shape[0]

            if mode == 1:
                mu = np.mean(x, axis=0)

            elif mode == 2:
                chosen_reps = np.random.choice(n_samples, K)
                mu = x[chosen_reps,:]
            else:
                ValueError('mode must be set to 1 or 2')

            std = np.std(x, axis=0)
            layer_gmm_params[gmm_layer_name].update({'mu': [mu]})
            layer_gmm_params[gmm_layer_name].update({'std': [std]})

    return layer_gmm_params

def calc_clusters_density_hist(n_explained_clusters, n_explaining_clusters, explained_clusters, explaining_clusters):
    """Counts the histogram of each appearance in depended clusters that associate with given clusters
    Args:
        n_explained_clusters(int) - the number of clusters layer L, that associates with them, contains.
        n_explaining_clusters(int) - the number of clusters layer L-1, that associates with them, contains.
        explained_clusters(int np array) - a vector contains the cluster index of a specific spatial location example across
                                       the batch. size (batch,1)
        explaining_clusters(int np array) - a tensor contains the cluster's index of the spatial examples that belong to
                                          the RF of the spatial example location in given_clusters vec according to
                                          layer L-1. size (batch, h, w) where h and w represents the size of the RF.
    """

    hist_mat = np.zeros((n_explained_clusters, n_explaining_clusters))

    for explained_k in range(n_explained_clusters):
        args = np.where(explained_clusters == explained_k)[0]
        corr_explaining_clusters = explaining_clusters[args]
        unique, counts = np.unique(corr_explaining_clusters, return_counts=True)

        hist_mat[explained_k, unique] = counts

    return hist_mat

def spatial_pairwise_hist(rf, explained_layer_name, explaining_layer_name, explained_gmm_preds, explaining_gmm_preds):
    """
    Calculate the p(h(L-1) = i | h(L) = j) where h is the cluster index in layer L.

    Args:
        rf (RF obj) - an RF object as define in receptivefield.py file.
        gmm_model(gmm_classification_model obj) - as given by gmm_classification_model_v3.py
        explained_layer_name(string) - the given layer name L (must be a GMM layer).
        explaining_layer_name (string) - (optional)the target layer name L-1 (must be a GMM layer).
        data(np array) - data to inference. size (batch, height, width, channels)
    Return:
         clusters_prob_mat (np array) - size (n_clusters_L-1, n_clusters_L). The value in each cell (i,j) represents the
                                        probability of a sample belonging to cluster i in layer 'target_layer_name',
                                        given that it was associated to cluster j in layer 'given_layer_name'.
    """

    explained_k_mat = get_most_likely_clusters(explained_gmm_preds)
    explaining_k_mat = get_most_likely_clusters(explaining_gmm_preds)

    explained_tens_shape = np.shape(explained_gmm_preds)
    H_dim = explained_tens_shape[1]
    W_dim = explained_tens_shape[2]

    n_explained_clusters = explained_tens_shape[-1]
    n_explaining_clusters = np.shape(explaining_gmm_preds)[-1]

    tot_clusters_hist = np.zeros((n_explained_clusters, n_explaining_clusters))

    for h in range(H_dim):
        for w in range(W_dim):
            size, center, UL_pos = rf.target_neuron_rf(explained_layer_name, [h, w],
                                                       rf_layer_name=explaining_layer_name,
                                                       return_upper_left_pos=True)
            upper_left_row = UL_pos[0]
            upper_left_col = UL_pos[1]

            rf_target_batch = explaining_k_mat[:, upper_left_row:upper_left_row+size[0],
                                               upper_left_col:upper_left_col+size[1]]
            given_clusters_vec = explained_k_mat[:,h,w]

            clusters_hist = calc_clusters_density_hist(n_explained_clusters, n_explaining_clusters,
                                                       given_clusters_vec, rf_target_batch)

            tot_clusters_hist += clusters_hist

    return tot_clusters_hist

def get_class_connections_dict_CP(class_index, appearance_hist_dict, clusters_stats,
                                  layers_to_inference, gmm_model, n_nodes=2, image_inference=False, image=None):
    """ Create nodes and edges for the inference tree according to class inference or image & class inference.

    Args:
        class_index (int) - the class index to be inference.
        appearance_hist_dict (dict) - a dictionary of co-occurrence matrices. Each key is a string from the format
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


    """
    connections_dict = {}
    image_locations_dict = {}

    explained_layer_name = 'gmm_classification'
    rep_clusters = [class_index]

    for explaining_layer_name in reversed(layers_to_inference):
        pairwise_name = explained_layer_name+'-'+explaining_layer_name

        gmm_explaining_layer_name = gmm_model.get_correspond_gmm_layer(explaining_layer_name)

        tot_clusters_hist = appearance_hist_dict[pairwise_name]
        empty_args = []
        empty_args = np.asarray(empty_args)

        for c in clusters_stats[gmm_explaining_layer_name]:
            if clusters_stats[gmm_explaining_layer_name][c] is None\
               or clusters_stats[gmm_explaining_layer_name][c]['fullness_percent'] < 0.06:
                c_ind = int(c) - 1
                empty_args = np.append(empty_args, c_ind)
        # P(h|h')
        sum = np.sum(tot_clusters_hist, axis=0)
        # empty_args = np.where(explaining_hist < (n_samples/n_explaining_clusters)*0.5)[0]
        sum[empty_args.astype('int')] = np.inf
        clusters_prob_mat = np.nan_to_num(np.round(tot_clusters_hist/sum, 5))

        # P(h'|h)
        # sum = np.sum(tot_clusters_hist, axis=1)
        # # sum = np.repeat(sum, np.shape(tot_clusters_hist)[1])
        # clusters_prob_mat = np.transpose(np.nan_to_num(np.round(np.transpose(tot_clusters_hist)/sum, 5)))
        # clusters_prob_mat[:, empty_args.astype('int')] = 0

        new_rep_clusters_prob = clusters_prob_mat[rep_clusters,:]
        new_clusters_power = np.sum(new_rep_clusters_prob, axis=0)
        subset_S = np.where(new_clusters_power > 0)[0]
        subset_S_power = new_clusters_power[subset_S]

        if image_inference:
            temp_image_locations_dict = {}
            if image is None:
                ValueError('If image_inference field is True, image_index is a mandatory require')
            else:
                corr_activation_layer = gmm_model.gmm_activations_dict[gmm_explaining_layer_name]
                img_pred = get_layer_output(keras_model=gmm_model.keras_model, layer_name=corr_activation_layer,
                                            input_data=image)
                H_dim = img_pred.shape[1]
                W_dim = img_pred.shape[2]
                K_dim = img_pred.shape[3]

                img_pred = np.reshape(img_pred, (H_dim*W_dim, K_dim))
                img_k = np.argmax(img_pred, axis=1)
                img_values = np.max(img_pred, axis=1)

                img_k_unique, img_k_counts = np.unique(img_k, return_counts=True)
                common_k = np.intersect1d(img_k_unique, subset_S)

                # tot_img_k_power = np.zeros(len(img_k_unique))
                comm_k_img_power = np.zeros(len(common_k))

                # for i, uni_k in enumerate(np.nditer(img_k_unique)):
                #     k_val = img_values[np.where(img_k == uni_k)[0]]
                #     k_val = np.sum(k_val)
                #
                #     tot_img_k_power[i] = k_val

                for i, uni_k in enumerate(np.nditer(common_k)):
                    location_inds = np.where(img_k == uni_k)[0]

                    xPos = np.remainder(location_inds, W_dim)
                    yPos = np.floor_divide(location_inds, W_dim)

                    uni_k_key = int(uni_k)
                    temp_image_locations_dict[uni_k_key] = {}
                    temp_image_locations_dict[uni_k_key]['xPos'] = xPos
                    temp_image_locations_dict[uni_k_key]['yPos'] = yPos

                    k_val1 = img_values[location_inds]
                    k_val1 = np.sum(k_val1)
                    # Stat 2
                    comm_k_img_power[i] = k_val1
                # Stat 1
                comm_k_explained_power = new_clusters_power[common_k]

                ord_com_k_img_p = np.argsort(-comm_k_img_power)
                ord_com_k_exp_p = np.argsort(-comm_k_explained_power)

                com_k_img_point = np.argsort(ord_com_k_img_p)
                com_k_exp_point = np.argsort(ord_com_k_exp_p)

                tot_com_k_point = com_k_exp_point + com_k_img_point
                new_rep_clusters = common_k[np.argsort(tot_com_k_point)]

        else:
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
                # new_rep_clusters_key.append(explaining_layer_name + '_' + str(depended_cluster + 1))
                new_rep_clusters_key.update(
                    {explaining_layer_name + '_' + str(explaining_cluster + 1):
                     clusters_prob_mat[explained_cluster, explaining_cluster]})

            connections_dict[cluster_key] = copy.deepcopy(new_rep_clusters_key)

        if image_inference:
            image_locations_dict[explaining_layer_name] = {}

            for rep_cluster in new_rep_clusters:
                image_locations_dict[explaining_layer_name][rep_cluster] = temp_image_locations_dict[rep_cluster]

    #   Update dependencies
        explained_layer_name = explaining_layer_name
        # rep_clusters = np.unique(new_rep_clusters[:,rep_clusters])
        rep_clusters = new_rep_clusters

    # set the first layer as nodes without connections
    for given_cluster in rep_clusters:
        cluster_key = explained_layer_name + '_' + str(given_cluster+1)
        connections_dict[cluster_key] = []

    if image_inference:
        return connections_dict, image_locations_dict
    else:
        return connections_dict

def create_cluster_stats(labels, clusters_rep, n_samples):
    """returns a dictionary contains the following fields for a certain layer:
        cluster_i: class_hist - a class histogram of examples associate with cluster i.
                   fullness - percentage of examples that have p(h|x) > 0.5
    """
    if len(np.shape(labels)) != 1:
        arg_labels = np.argmax(labels, axis=1)

    classes = labels.shape[1]
    clusters_stats = {}
    for c in clusters_rep:

        if clusters_rep[c] is not None:
            clusters_stats[c] = {}
            cluster_labels = arg_labels[clusters_rep[c]]

            unique_labels, counts_labels = np.unique(cluster_labels, return_counts=True)
            nc_samples = np.sum(counts_labels)
            class_hist = np.zeros(classes)
            class_hist[unique_labels] = counts_labels
            dom_class = np.argmax(class_hist)

            clusters_stats[c]['fullness_percent'] = round((nc_samples / n_samples)*100, 3)
            clusters_stats[c]['fullness_frac'] = str(nc_samples)+'/'+str(n_samples)
            # clusters_stats[c]['class_hist'] = class_hist.tolist()
            clusters_stats[c]['dominant_class'] = int(dom_class)
            clusters_stats[c]['dominant_class_percent'] = round((class_hist[dom_class] / nc_samples) * 100, 3)

        else:
            clusters_stats[c] = None
    return clusters_stats

def find_class_representatives(clusters_stats):
    """for each class find the M dominant class clusters. this calculation is for a specific layer
    Args -
        clusters_stats(dict) - as returns by 'create_cluster_stats' func.
    """
    classes_dict = {}

    for k in clusters_stats:
        if clusters_stats[k] is not None:
            dominant_class = clusters_stats[k]['dominant_class']

            if dominant_class not in classes_dict:
                classes_dict[dominant_class] = {'clusters_rep': np.asarray(k),
                                                'dominant_class_percent': np.asarray(clusters_stats[k]['dominant_class_percent'])}

            else:
                classes_dict[dominant_class]['clusters_rep'] = np.append(classes_dict[dominant_class]['clusters_rep'], k)
                classes_dict[dominant_class]['dominant_class_percent'] = np.append(
                    classes_dict[dominant_class]['dominant_class_percent'], clusters_stats[k]['dominant_class_percent'])

    for c in classes_dict:
        sorted_args = np.argsort(classes_dict[c]['dominant_class_percent'])
        sorted_args = np.flip(sorted_args)

        if len(sorted_args) > 1:
            classes_dict[c]['dominant_class_percent'] = classes_dict[c]['dominant_class_percent'][sorted_args].tolist()
            classes_dict[c]['clusters_rep'] = classes_dict[c]['clusters_rep'][sorted_args].tolist()
        else:
            classes_dict[c]['dominant_class_percent'] = float(classes_dict[c]['dominant_class_percent'])
            classes_dict[c]['clusters_rep'] = int(classes_dict[c]['clusters_rep'])

    return classes_dict














