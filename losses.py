from keras import backend as K

def build_gmm_loss(hyper_lambda):
    def gmm_loss(y_true, y_pred):
        x_dim = K.shape(y_pred)

        if y_pred.get_shape().ndims == 4:
            pool_dim = K.stack([x_dim[0], x_dim[1], x_dim[2], x_dim[3]])
            n_samples = pool_dim[0]*pool_dim[1]*pool_dim[2]

        elif y_pred.get_shape().ndims == 2:
            pool_dim = K.stack([x_dim[0], x_dim[1]])
            n_samples = pool_dim[0]

        n_gaussians = y_pred.get_shape()[-1]
        s_k = K.reshape(y_pred, (-1, n_gaussians))
        s_max = K.max(s_k, axis=1, keepdims=True)
        s_max_rep = K.repeat_elements(s_max, n_gaussians, axis=1)
        l_f = K.logsumexp((s_k - s_max_rep), axis=1, keepdims=True)
        p = s_max + l_f
        sum_log_prob = K.sum(p)
        float_ground = K.cast((n_samples * n_gaussians), dtype='float32')
        norm_log_prob = sum_log_prob / float_ground

        return -(hyper_lambda * norm_log_prob)
    return gmm_loss