import numpy as np
from keras.models import Model


class ReceptiveField():
    def __init__(self, model=None):
        self.model = model
        # Auxiliary dictionary to describe the network graph
        self.network_dict = {'input_layers_of': {}, 'output_layers_of': {}, 'output_tensor_of': {},
                             'predecessor_convs': {}}
        self.size = {}
        self.induced_stride = {}
        self.kernel = {}
        self.add_conv_dict = {}

        if not self.model:
            raise ValueError('keras model is a mandatory demand for ReceptiveField object')

        self._create_network_dict()
        self._calc_size()
        self.conv_to_conv_size = {}

    def _create_network_dict(self):
        """Returns a dict of kernel, stride and padding params of each layer
        """
        # Set the input count_layers of each layer
        for layer in self.model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in self.network_dict['input_layers_of']:
                    self.network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
                else:
                    self.network_dict['input_layers_of'][layer_name].append(layer.name)

                if layer.name not in self.network_dict['output_layers_of']:
                    self.network_dict['output_layers_of'].update(
                        {layer.name: [layer_name]})
                else:
                    self.network_dict['output_layers_of'][layer.name].append(layer_name)

        # Set the output tensor of the input layer
        self.network_dict['output_tensor_of'].update(
            {self.model.layers[0].name: self.model.input})

        # Iterate over all count_layers after the input
        for layer in self.model.layers[1:]:

            # Determine input tensors
            layer_input = [self.network_dict['output_tensor_of'][layer_aux]
                           for layer_aux in self.network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted
            # layer)
            self.network_dict['output_tensor_of'].update({layer.name: x})

        # Saving subsequent tree of conv or pooling count_layers.
        for layer_name in self.network_dict['output_layers_of']:
            layer = self.model.get_layer(layer_name)
            layer_type = type(layer).__name__

            if layer_type == 'Conv2D' or layer_type == 'AveragePooling2D' or\
                            layer_type == 'MaxPooling2D' or layer_type == 'Add':
                # split_or_finish = False
                found_conv = False
                post_convs = []

                post_layers_name = self.network_dict['input_layers_of'][layer_name]

                while len(post_layers_name) == 1 and not found_conv:
                    post_layer_name = post_layers_name[0]
                    post_layer = self.model.get_layer(post_layer_name)
                    post_layer_type = type(post_layer).__name__

                    if post_layer_type == 'InputLayer':
                        found_conv = True
                        post_convs.append(None)
                        break

                    if post_layer_type == 'Conv2D' or post_layer_type == 'AveragePooling2D' \
                            or post_layer_type == 'MaxPooling2D':
                        found_conv = True
                        post_convs.append(post_layer_name)

                    post_layers_name = self.network_dict['input_layers_of'][post_layer_name]

                if not found_conv:
                    for post_layer_name in post_layers_name:
                        split_or_finish = False
                        post_layer = self.model.get_layer(post_layer_name)
                        post_layer_type = type(post_layer).__name__

                        if post_layer_type == 'Conv2D' or post_layer_type == 'AveragePooling2D' \
                                or post_layer_type == 'MaxPooling2D':
                            post_convs.append(post_layer_name)

                        else:
                            pre_layers_name = self.network_dict['output_layers_of'][post_layer_name]
                            post2_layers_name = self.network_dict['input_layers_of'][post_layer_name]

                            while not split_or_finish:
                                if len(pre_layers_name) > 1:
                                    split_or_finish = True

                                else:
                                    post2_layer_name = post2_layers_name[0]
                                    post2_layer = self.model.get_layer(post2_layer_name)
                                    post2_layer_type = type(post2_layer).__name__

                                    if post2_layer_type == 'Conv2D' or post2_layer_type == 'AveragePooling2D' \
                                            or post2_layer_type == 'MaxPooling2D':
                                        post_convs.append(post2_layer_name)
                                        split_or_finish = True

                                    else:
                                        post2_layers_name = self.network_dict['input_layers_of'][post2_layer_name]

                if layer_type == 'Add':
                    for conv_layer in post_convs:
                        if layer.name not in self.add_conv_dict:
                            self.add_conv_dict.update(
                                {layer.name: [conv_layer]})
                        else:
                            self.add_conv_dict[layer.name].append(conv_layer)
                else:
                    for conv_layer in post_convs:
                        if layer.name not in self.network_dict['predecessor_convs']:
                            self.network_dict['predecessor_convs'].update(
                                {layer.name: [conv_layer]})
                        else:
                            self.network_dict['predecessor_convs'][layer.name].append(conv_layer)

    def _calc_size(self):
        """calc the effective receptive field size of a certain layer on the input
        Args:
            model - a keras model
            layer_number (int) optional - the layer number to calculate the RF size for
            layer_name (string) optional - the layer name to calculate the RF size for

            Note: one of the the fields 'layer_number' or 'layer_name' must be given.

        Returns:
            rf_size(int)- the rf size in pixels of the input.

            """
        for conv_name in self.network_dict['predecessor_convs']:
            layer = self.model.get_layer(conv_name)
            layer_type = type(layer).__name__
            calc_size = True

            if layer_type == 'Conv2D':
                kernel_size = layer.kernel_size
            elif layer_type == 'AveragePooling2D' \
                    or layer_type == 'MaxPooling2D':
                kernel_size = layer.pool_size
            else:
                raise ValueError('The ReceptiveField calculations support Conv2D, AveragePooling2D '
                                 'and MaxPooling2D keras layers only')

            if calc_size:
                self.kernel[conv_name] = kernel_size

                strides = layer.strides
                pre_convs = self.network_dict['predecessor_convs'][conv_name]

                if not pre_convs[0]:
                    induced_stride = strides
                    r_x = 1 + (kernel_size[0] - 1) * 1
                    r_y = 1 + (kernel_size[1] - 1) * 1

                else:
                    if len(pre_convs) == 1:
                        pre_conv = pre_convs[0]

                    else:
                        max_rf_size = (0,0)
                        max_pre_conv = None

                        for pre_conv in pre_convs:
                            temp_rf_size = self.size[pre_conv]

                            if temp_rf_size > max_rf_size:
                                max_rf_size = temp_rf_size
                                max_pre_conv = pre_conv

                        pre_conv = max_pre_conv

                    induced_stride = self.induced_stride[pre_conv]
                    r_x_prev = self.size[pre_conv][0]
                    r_y_prev = self.size[pre_conv][1]
                    r_x = r_x_prev + (kernel_size[0] - 1) * induced_stride[0]
                    r_y = r_y_prev + (kernel_size[1] - 1) * induced_stride[1]

                    induced_stride = tuple(np.asarray(strides)*np.asarray(induced_stride))

                self.induced_stride.update({conv_name: induced_stride})
                self.size.update({conv_name: (r_x, r_y)})

        for add_layer, convs_layers in self.add_conv_dict.items():
            max_rf_size = (0, 0)
            max_pre_conv = None
            for pre_conv in convs_layers:
                temp_rf_size = self.size[pre_conv]
                if temp_rf_size > max_rf_size:
                    max_rf_size = temp_rf_size
                    max_pre_conv = pre_conv
            self.size.update({add_layer: max_rf_size})
            self.add_conv_dict.update({add_layer: max_pre_conv})

    def _calc_partial_rf_size(self, rf_layer_name, target_layer_name):
        """calc the effective receptive field size of a certain layer on the input
        Args:
            model - a keras model
            layer_number (int) optional - the layer number to calculate the RF size for
            layer_name (string) optional - the layer name to calculate the RF size for

            Note: one of the the fields 'layer_number' or 'layer_name' must be given.

        Returns:
            rf_size(int)- the rf size in pixels of the input.

            """
        size = {}
        induced_stride_dict = {}
        stop_flag = False
        start_flag = False
        first_entrace = True

        relevant_layers = [target_layer_name]
        curr_layer = target_layer_name
        key_name = target_layer_name + '-' + rf_layer_name

        while not stop_flag:
            curr_layers = self.network_dict['predecessor_convs'][curr_layer]

            if len(curr_layers) == 1:
                curr_layer = curr_layers[0]
                if curr_layer == rf_layer_name:
                    stop_flag = True
            else:
                for curr_layer in curr_layers:
                    if curr_layer == rf_layer_name:
                        stop_flag = True
                        break
            relevant_layers.insert(0, curr_layer)

            if not stop_flag and curr_layer is None:
                raise ValueError(f'Could not calculate {target_layer_name} receptive field for layer {rf_layer_name}')

        # while not stop_flag:
        for conv_name in relevant_layers:

            if conv_name == rf_layer_name:
                start_flag = True
            # if conv_name == target_layer_name:
            #     stop_flag = True

            if start_flag:
                layer = self.model.get_layer(conv_name)
                layer_type = type(layer).__name__

                if layer_type == 'Conv2D':
                    kernel_size = layer.kernel_size
                elif layer_type == 'AveragePooling2D' or layer_type == 'MaxPooling2D':
                    kernel_size = layer.pool_size
                else:
                    raise ValueError('The ReceptiveField calculations support Conv2D,'
                                     ' AveragePooling2D and MaxPooling2D keras layers only')

                strides = layer.strides
                pre_convs = self.network_dict['predecessor_convs'][conv_name]

                if first_entrace:
                    induced_stride = strides
                    r_x = 1
                    r_y = 1
                    first_entrace = False
                else:
                    if len(pre_convs) == 1:
                        pre_conv = pre_convs[0]
                    else:
                        max_rf_size = (0,0)
                        max_pre_conv = None
                        for pre_conv in pre_convs:
                            if pre_conv in relevant_layers:
                                temp_rf_size = size[pre_conv]

                                if temp_rf_size > max_rf_size:
                                    max_rf_size = temp_rf_size
                                    max_pre_conv = pre_conv
                        pre_conv = max_pre_conv

                    induced_stride = induced_stride_dict[pre_conv]
                    r_x_prev = size[pre_conv][0]
                    r_y_prev = size[pre_conv][1]
                    r_x = r_x_prev + (kernel_size[0] - 1) * induced_stride[0]
                    r_y = r_y_prev + (kernel_size[1] - 1) * induced_stride[1]
                    induced_stride = tuple(np.asarray(strides)*np.asarray(induced_stride))

                induced_stride_dict.update({conv_name: induced_stride})
                size.update({conv_name: (r_x, r_y)})

        self.conv_to_conv_size.update({key_name: size[target_layer_name]})
        return size[target_layer_name]

    @staticmethod
    def _position_formula(row_ind, column_ind, kernel_size, strides, padding):

        if strides[0] == 1:
            row_l_prev = row_ind
        else:
            row_l_prev = strides[0]*row_ind + np.floor_divide(kernel_size[0], 2)
        if strides[1] == 1:
            column_l_prev = column_ind
        else:
            column_l_prev = strides[1]*column_ind + np.floor_divide(kernel_size[1], 2)

        if padding == 'valid':
            row_l_prev = row_l_prev + np.floor_divide(kernel_size[0], 2)
            column_l_prev = column_l_prev + np.floor_divide(kernel_size[1], 2)

        return row_l_prev, column_l_prev

    def _correct_pos(self, layer_name, target_input_center, rf_layer_name=None, return_UL_pos=False,
                     return_origin_center=False, return_origin_size=False):

        if 'input' not in rf_layer_name:
            input_shape = self.model.get_layer(name=rf_layer_name).input_shape
            conv2conv_name = layer_name +'-'+ rf_layer_name
            if conv2conv_name not in self.conv_to_conv_size:
                size = self._calc_partial_rf_size(rf_layer_name=rf_layer_name, target_layer_name=layer_name)
            else:
                size = self.conv_to_conv_size[conv2conv_name]
        else:
            input_shape = self.model.input_shape
            size = self.size[layer_name]

        upper_left_row = target_input_center[0] - np.floor_divide(size[0], 2)
        upper_left_col = target_input_center[1] - np.floor_divide(size[1], 2)

        target_rows = np.arange(upper_left_row, (upper_left_row + size[0]))
        target_cols = np.arange(upper_left_col, (upper_left_col + size[1]))

        input_rows = np.arange(input_shape[1])
        input_cols = np.arange(input_shape[2])

        common_rows = np.intersect1d(target_rows, input_rows)
        common_cols = np.intersect1d(target_cols, input_cols)

        upper_left = (np.min(common_rows), np.min(common_cols))
        bottom_right = (np.max(common_rows), np.max(common_cols))

        center = (int((upper_left[0] + bottom_right[0])/2), int((upper_left[1] + bottom_right[1])/2))
        new_size = (bottom_right[0] - upper_left[0] + 1, bottom_right[1] - upper_left[1] + 1)

        returns_vec = np.asanyarray([new_size, center])

        if return_UL_pos:
            returns_vec = np.concatenate((returns_vec, np.array([upper_left])), axis=0)
        if return_origin_center:
            returns_vec = np.concatenate((returns_vec, np.array([target_input_center])), axis=0)
        if return_origin_size:
            returns_vec = np.concatenate((returns_vec, np.array([size])), axis=0)

        return returns_vec

    def target_neuron_rf(self, layer_name, target_neuron, rf_layer_name=None, return_origin_center=False,
                         return_upper_left_pos=False, return_origin_size=False):

        row_l = target_neuron[0]
        column_l = target_neuron[1]

        if 'add' in layer_name:
            max_pre_conv = self.add_conv_dict[layer_name]
            layer_name = max_pre_conv
            layer = self.model.get_layer(layer_name)

        else:
            layer = self.model.get_layer(layer_name)

        if 'add' in rf_layer_name:
            rf_layer_name = self.add_conv_dict[rf_layer_name]

        # if rf_layer_name is not None the calculation will stop when we get to rf_layer
        stop_flag = False

        if type(layer).__name__ != 'Conv2D':
            raise ValueError('Receptive Field is calculated for Conv2D keras count_layers only')

        kernel_size = layer.kernel_size
        strides = layer.strides
        padding = layer.padding

        row_l, column_l = self._position_formula(row_l, column_l, kernel_size, strides, padding)
        pre_convs = self.network_dict['predecessor_convs'][layer_name]

        while pre_convs[0] and not stop_flag:

            if len(pre_convs) == 1:
                pre_conv = pre_convs[0]

                if pre_conv == rf_layer_name:
                    stop_flag = True
            else:
                max_rf_size = (0, 0)
                max_pre_conv = None

                for pre_conv in pre_convs:
                    if pre_conv == rf_layer_name:
                        stop_flag = True

                    temp_rf_size = self.size[pre_conv]

                    if temp_rf_size > max_rf_size:
                        max_rf_size = temp_rf_size
                        max_pre_conv = pre_conv

                pre_conv = max_pre_conv

            if not stop_flag:
                layer = self.model.get_layer(pre_conv)
                layer_type = type(layer).__name__

                if layer_type == 'Conv2D':
                    kernel_size = layer.kernel_size
                elif layer_type == 'AveragePooling2D' or layer_type == 'MaxPooling2D':
                    kernel_size = layer.pool_size
                else:
                    raise ValueError('The ReceptiveField calculations support Conv2D, AveragePooling2D'
                                     ' or MaxPooling2D keras layers only')

                strides = layer.strides
                # padding = layer.padding
                row_l, column_l = self._position_formula(row_l, column_l, kernel_size, strides, 'same')
                pre_convs = self.network_dict['predecessor_convs'][pre_conv]

        if 'input' not in rf_layer_name and not stop_flag:
            raise ValueError(rf_layer_name, ' where not found')

        # Adjusts the RF size to the input size
        return self._correct_pos(layer_name=layer_name, target_input_center=(row_l, column_l),
                                 rf_layer_name=rf_layer_name,
                                 return_UL_pos=return_upper_left_pos,
                                 return_origin_center=return_origin_center,
                                 return_origin_size=return_origin_size)

    @staticmethod
    def _set_weights_and_bias(keras_model, weight_int=1, bias_int=0):

        for layer in keras_model.layers:
            if type(layer).__name__ == 'Conv2D':
                weights = layer.get_weights()
                weights[0] = np.ones_like(weights[0]) * weight_int
                weights[1] = np.ones_like(weights[1]) * bias_int
                layer.set_weights(weights)

        return keras_model

    @staticmethod
    def _calc_influence_input_neuron(keras_model, input_data):

        intermediate_output = keras_model.predict(input_data)
        single_map_output = intermediate_output[0, :, :, 0]
        binary_output = np.zeros_like(single_map_output)
        binary_output[np.where(single_map_output != 0)] = 1

        return binary_output

    @staticmethod
    def _find_rectangle(image):
        ones_indices = np.where(image != 0)

        column_size = ones_indices[1][-1] - ones_indices[1][0] + 1
        row_size = ones_indices[0][-1] - ones_indices[0][0] + 1

        column_center = int((ones_indices[1][-1] + ones_indices[1][0]) / 2)
        row_center = int((ones_indices[0][-1] + ones_indices[0][0]) / 2)

        return (row_size, column_size), (row_center, column_center)