import tensorflow as tf

def exponential_decay(initial_value, decay_rate, decay_steps, step):
        return initial_value * decay_rate ** (step / decay_steps) 


def gumbel_softmax(logits, tau, axis=-1):
    shape = tf.keras.backend.int_shape(logits)
    
    # Gumbel(0, 1)
    if len(shape) == 1:
        gumbels = tf.math.log(tf.random.gamma(shape, 1))
    else:
        gumbels = tf.math.log(
            tf.random.gamma(shape[:-1], [1 for _ in range(shape[-1])])
        )
        
    # Gumbel(logits, tau)
    gumbels = (logits + gumbels) / tau
    
    y_soft = tf.nn.softmax(gumbels, axis=axis)
    
    return y_soft

def dense_layer_flops(input_size, output_size):
     return (2 * input_size * output_size) + output_size