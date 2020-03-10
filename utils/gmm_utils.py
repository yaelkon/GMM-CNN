import numpy as np
from tensorflow import set_random_seed

np.random.seed(101)
set_random_seed(101)


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
        else:
            raise ValueError(f'The modeled layer tensor shape must be either 2 or 4, but got: {len(tens_shape)}')

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

                if len(tens_shape) == 4:
                    image_inds = np.ceil((k_indices + 1) / (H_dim * W_dim)).astype('int32') - 1

                elif len(tens_shape) == 2:
                    image_inds = k_indices
                else:
                    raise ValueError( 'the output shape must be either 4 for convolutional layer or'
                                      ' 2 for dense layer, but got: {len(shape)}' )

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


def get_cluster_reps(clusters_dict, H_dim=None, W_dim=None, n_reps=10):
    dic = {}
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

            if llr_exists and k_values[0] == 1:
                threshold_ind = np.where(k_values == 1)[0][-1]
                one_value_indices = k_indices[:(threshold_ind + 1)]
                llr_value = k_llr[:(threshold_ind + 1)]
                argsort = np.argsort(-llr_value)
                one_value_indices = one_value_indices[argsort]
                k_indices[:(threshold_ind + 1)] = one_value_indices
                k_llr[:(threshold_ind + 1)] = llr_value[argsort]

            new_n_rep = n_reps
            if len(k_indices) < n_reps:
                new_n_rep = len(k_indices)
            k_indices = k_indices[:new_n_rep]
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

            dic[cluster] = {}
            dic[cluster]['spatial_location'] = {'col': np.transpose(xPos).tolist()}
            dic[cluster]['spatial_location'].update({'row': np.transpose(yPos).tolist()})
            dic[cluster]['image'] = np.transpose(image_inds).tolist()

    return dic


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

            clusters_stats[c]['fullness_percent'] = round((nc_samples / n_samples) * 100, 3)
            clusters_stats[c]['fullness_frac'] = str(nc_samples)+'/'+str(n_samples)
            clusters_stats[c]['dominant_class'] = int(dom_class)
            clusters_stats[c]['dominant_class_percent'] = round((class_hist[dom_class] / nc_samples) * 100, 3)

        else:
            clusters_stats[c] = None
    return clusters_stats







