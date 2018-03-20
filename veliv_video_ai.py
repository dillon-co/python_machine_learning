import numpy as np
import tensorflow as tf

# video_inputs = np.array(3, 4)
video_batch_size = 6
num_video_input_paramaters = 3
num_video_output_paramaters =

x = tf.placeholder('float', [video_batch_size, num_video_paramaters])
y = tf.placeholder('float', [num__videos, num_output_paramaters, num_videos])


def policy_based_neural_net(x):
    
