from graphviz import Digraph
import os
import numpy as np
import time


class Inference_Graph:

    def __init__(self, name=None,
                 head_class_name=None,
                 head_class_index=None,
                 connections_dict=None,
                 imgs_path=None,
                 layers_clusters_dict=None,
                 n_nodes=3,
                 edge_label_font_size='18',
                 header_font_size='50',
                 heat_map_connections=False,
                 node_type='patches',
                 level_gap=1,
                 saving_format='png',
                 saving_dir='/tmp/',
                 heat_map_path=None,
                 image_inference=False):

        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        self.head_class_name = head_class_name
        self.head_class_index = head_class_index
        self.connections_dict = connections_dict
        self.imgs_path = imgs_path
        self.node_type = node_type
        self.gap = level_gap
        self.saving_format = saving_format
        self.saving_dir = saving_dir
        self.heat_map_dir = heat_map_path
        self.layers_clusters_dict = layers_clusters_dict
        self.heat_map_connections = heat_map_connections
        self.n_nodes = n_nodes
        self.image_inf = image_inference
        self.edge_label_font_size = edge_label_font_size

        self.graph = Digraph(name=name, format=saving_format)
        self.graph.graph_attr.update({'label': 'Class ' + head_class_name})
        self.graph.graph_attr.update({'fontsize': header_font_size})
        self.graph.graph_attr.update({'labelloc': 't'})
        self.graph.graph_attr.update({'splines': 'false', 'center': 'true',
                                      'nodesep': "0.0", 'ranksep': "0.0", 'rsatio': 'compress'
                                      , 'outputorder': "nodesfirst"
                                      })
        self.graph.edge_attr.update({'minlen': '1'})
        self._build_tree()

    def _build_tree(self):

        connections_dict = self.connections_dict

        if self.gap > 1:
            self.reduced_connections_dict = {}
            nodes = self.connections_dict['class']
            self.reduced_connections_dict['class'] = nodes
            reached_last_level = False

            while not reached_last_level:
                temp_nodes = []

                for node in nodes:
                    temp_node = node
                    for i in range(self.gap - 1):
                        if temp_node in self.connections_dict:
                            temp_node = self.connections_dict[temp_node][0]
                        else:
                            reached_last_level = True
                            break

                    if reached_last_level:
                        break
                    else:
                        gap_nodes = self.connections_dict[temp_node]
                        temp_nodes += gap_nodes
                        self.reduced_connections_dict.update({node: gap_nodes})

                nodes = np.unique(temp_nodes)

                if nodes[0] not in self.connections_dict:
                    reached_last_level = True

            # Set the first layer as nodes without connections
            for node in nodes:
                self.reduced_connections_dict.update({node: []})
            connections_dict = self.reduced_connections_dict
        levels_dict = {}
        for node in connections_dict:
            if node == 'class':
                if self.image_inf:
                    cluster_name = 'classification_cluster_' + str(self.head_class_index + 1) + '.png'
                    img_path = os.path.join(*[self.imgs_path, cluster_name])
                else:
                    cluster_name = 'cluster_' + str(self.head_class_index + 1) + '.png'
                    img_path = os.path.join(*[self.imgs_path, 'classification', cluster_name])
            else:
                layer_name = node.split('_')[0]
                for i in range(len(node.split('_')) - 2):
                    layer_name = layer_name + '_' + node.split('_')[i + 1]

                cluster_num = node.split('_')[-1]

                if self.image_inf:
                    if 'fc' in layer_name:
                        cluster_name = layer_name + '_cluster_' + cluster_num + '.png'
                    else:
                        cluster_name = layer_name + '_cluster_' + cluster_num + '_' + self.node_type[layer_name] + '.png'
                    img_path = os.path.join(*[self.imgs_path, cluster_name])
                else:
                    if 'fc' in layer_name:
                        cluster_name = 'cluster_' + cluster_num + '.png'
                    else:
                        cluster_name = 'cluster_' + cluster_num + '_' + self.node_type[layer_name] + '.png'
                    img_path = os.path.join(*[self.imgs_path, layer_name, cluster_name])

                if layer_name in levels_dict:
                    levels_dict[layer_name].append(node)
                else:
                    levels_dict[layer_name] = [node]

            self.graph.node(node, label="", shape='box', color='white', fontsize='0', image=img_path,
                            imagescale='height', margin='0.0')
        # set the layers to be at the same level in the graph
        for key in levels_dict:
            with self.graph.subgraph() as s:
                s.attr(rank='same')
                for n in levels_dict[key]:
                    s.node(n)
        for key, values in connections_dict.items():
            if values:
                for value in values:
                    if 'LR' in connections_dict[key][value]:
                        edge_strength = connections_dict[key][value]['LR']
                        label = str(connections_dict[key][value]['CN']) + ', ' + str(edge_strength)
                    else:
                        edge_strength = connections_dict[key][value]
                        label = str(edge_strength)

                    #     Weak connection
                    if edge_strength <= 0:
                        color = 'white'
                        penwidth = '0.0'
                    #     Medium connection
                    elif edge_strength <= 1:
                        color = 'black'
                        penwidth = '2.0'
                    #     Strong connection
                    else:
                        color = 'green'
                        penwidth = '3.0'

                    if key == 'class':
                        if penwidth == '0.0':
                            penwidth = '2.0'
                        if color == 'white':
                            color = 'black'
                        self.graph.edge(key, value, label=label, fontsize=self.edge_label_font_size,
                                        penwidth=penwidth, minlen="3.0", color=color, dir='back')

                    # For fc layers and configuration without heat maps, do not use heat maps.
                    elif 'fc' in key or 'fc' in value or not self.heat_map_connections:
                        self.graph.edge(key, value, label=label, fontsize=self.edge_label_font_size,
                                        penwidth=penwidth, color=color, dir='back')
                    elif color != 'white' and penwidth != '0.0':
                        connecting_node_name = key + '_' + value
                        heat_map_path = os.path.join(self.heat_map_dir, connecting_node_name +'.png')

                        self.graph.node(connecting_node_name, label="", shape='box', color='white', fontsize='0',
                                        labelloc='t',
                                        fixedsize='true', width='1.8', height='1.5', image=heat_map_path)
                        self.graph.edge(key, connecting_node_name,
                                        fontsize=self.edge_label_font_size,
                                        penwidth=penwidth, color=color, dir='back', minlen='3.0')
                        self.graph.edge(connecting_node_name, value, penwidth=penwidth,
                                        color=color, dir='none', len='0.5')
                    else:
                        self.graph.edge(key, value, fontsize='0', penwidth=penwidth, color=color, dir='none')

        file_name = self.graph.name + '_n=' + str(self.n_nodes) +\
                    '_' + time.strftime('%Y%m%d_%H%M%S') + '.gv'
        self.graph.render(filename=file_name, directory=self.saving_dir, view=False, cleanup=True, format='png')
