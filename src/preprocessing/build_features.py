import tensorflow as tf
from datasets.load_datasets import FrameGenerator

def train_test_split(data):
    n_frames = 10
    batch_size = 8

    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                        tf.TensorSpec(shape = (), dtype = tf.int16))

    train_ds = tf.data.Dataset.from_generator(FrameGenerator(data['train'], n_frames, training=True),
                                            output_signature = output_signature)


    # Batch the data
    train_ds = train_ds.batch(batch_size)

    val_ds = tf.data.Dataset.from_generator(FrameGenerator(data['val'], n_frames),
                                            output_signature = output_signature)
    val_ds = val_ds.batch(batch_size)

    test_ds = tf.data.Dataset.from_generator(FrameGenerator(data['test'], n_frames),
                                            output_signature = output_signature)

    test_ds = test_ds.batch(batch_size)
    return train_ds, test_ds, val_ds