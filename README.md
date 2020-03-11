# GMM-CNN
This code package implements the modeling of CNN layers activity with Gaussian mixture model and Inference Graphs visualization technique from the paper "Inference graphs for CNN Interpretation". 

## Requirements:
- Graphvis V
- Tensorflow
- Keras
- Numpy

## Instructions - work flow:
### GMM-CNN model training (main.py):
1. Specify the directory for saving the model and configuration
2. Specify the layers names as used in the CNN you wish to model.
3. Specify the number of Gaussian (clusters) for each layer (as the same order you did in 2.).
4. Choose between discriminative/generative for training method (as explained in the paper).
5. Run main.py. 

### Gathering clusters statistics (gather_clusters_stats.py):
1. Specify the directory you gave in 1. above.
2. Run gather_clusters_stats.py script.

### Creating clusters images (draw_clusters.py):
1. Specify the directory you gave in 1. above.
2. Insert the cluster visualiztion technique (rectangle/patches).
3. Run draw_clusters.py.

### Creating class infernce graphs (draw_class_inference_graph.py): /draw_image_inference_graph.py):
1. Specify the directory you gave in 1. above.
2. Specify the CNN layers name to visualize in the graph and the clusters visualiztion technique for each layer.
3. Specify the class name you want to build the graph for.
4. Run draw_class_inference_graph.py.

### Creating image infernce graphs (draw_image_inference_graph.py): 
1. Specify the directory you gave in 1. above.
2. Specify the CNN layers name to visualize in the graph and the clusters visualiztion technique for each layer.
3. Specify whther you want to analyze a well-classified images or wrongly-classified images.
4. Specify the number of images/graphs you want to build.
5. Run draw_image_inference_graph.py.
