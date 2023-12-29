import os
import zipfile
import numpy as numpy
import tensorflow as tf
from functools import partial
import PIL.Image
import urllib.request

def main():

    # URL link to inception zip file
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = '../data' # create dir for our data
    model_name = os.path.split(url)[-1]
    loc_zip_file = os.path.join(data_dir, model_name)

    if not os.path.exists(loc_zip_file):
        model_url = urllib.request.urlopen(url)

        with open(loc_zip_file, 'wb') as f:
            f.write(model_url.read())

        with zipfile.ZipFile(loc_zip_file, 'r') as zip_refer:
            zip_refer.extractall(data_dir)

    model_func = 'tensorflow_inception_graph.pb'

    # now we create a TF session and load the graph
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_func), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tensor_input = tf.placeholder(numpy.float32, name='input') # our input tensor
    imagenetwork_mean = 117.0
    tensor_preprocessed = tf.expand_dims(tensor_input - imagenetwork_mean, 0)
    tf.import_graph_def(graph_def, {'input': tensor_preprocessed})

    # define our layers
    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]

    # load layers into array
    feature_nums = [int(graph.get_tensor_by_name(name +':0').get_shape()[-1]) for name in layers]

    # debugging print statements:
    print(f'Number of layers: {len(layers)}')
    print(f'Feature channels: {sum(feature_nums)}')
    print(f'Feature numbers: {feature_nums}')

main()