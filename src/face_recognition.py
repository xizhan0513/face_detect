import numpy as np
import cv2
from sklearn import preprocessing
import os
from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
import copy

def t(pb_path, image):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
 
                input_image_tensor = sess.graph.get_tensor_by_name('img_inputs:0')
                #input_is_training_tensor = sess.graph.get_tensor_by_name('trainable_bn:0')
                output_tensor_name = sess.graph.get_tensor_by_name('outputs:0')
 
                out = sess.run(output_tensor_name, feed_dict={
                                 input_image_tensor:image
                                 })
    return out



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    image = cv2.imread('./zhongyuqing_test.jpg').astype(np.float32)	#(112,112,3)
    image = image[...,::-1] #rgb

     
    img1 = image.copy()
    img1 = img1.transpose(2,0,1)	#(3,112,112)
    img1 = img1[np.newaxis,:]	#(1,3,112,112)
    img1 -= 127.5
    img1 *= 0.0078125
    out1 = t('./mobilefacenet_v6.pb', img1)

    img2 = image.copy()
    img2 = np.fliplr(img2)	#flip horizintal
    img2 = img2.transpose(2,0,1)	#(3,112,112)
    img2 = img2[np.newaxis,:]	#(1,3,112,112)
    img2 -= 127.5
    img2 *= 0.0078125
    out2 = t('./mobilefacenet_v6.pb', img2)


    out = out1 + out2
    features = preprocessing.normalize(out)

    databank_h = np.load('./huanglei_rgb.npy')    
    distance = np.sum(np.square(np.subtract(features, databank_h)))
    print(distance)
