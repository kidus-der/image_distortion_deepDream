import os
import zipfile
import numpy as np
import tensorflow as tf
from functools import partial
import PIL.Image
import urllib.request
import matplotlib.pyplot as plt

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

    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    model_func = 'tensorflow_inception_graph.pb'

    # now we create a TF session and load the graph
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_func), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tensor_input = tf.placeholder(np.float32, name='input') # our input tensor
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

    # strip large constants from graph
    def strip_constants(graph_def, max_const_size=32):
        
        strip_def = tf.GraphDef()
        for n_0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n_0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
        return strip_def
    
    #rename node identifiers
    def rename_nodes(graph_def, rename_func):

        res_def = tf.GraphDef()
        for n_0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n_0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        return res_def
    
    def show_img(img):
        
        img = np.uint8(np.clip(img, 0, 1)*255)
        plt.imshow(img)
        plt.show()

    def norm_visualization(img, s=0.1):

        return (img - img.mean())/max(a.std(), 1e-4)*s + 0.5
    
    def output_tensor(layer):
        return graph.get_tensor_by_name("import/%s:0"%layer)
    
    