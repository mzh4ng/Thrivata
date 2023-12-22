import tensorflow as tf
import wget
import tarfile
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

def load_movinet(config):
  """Loads the pre-trained MoviNet model"""
  tf.keras.backend.clear_session()

  backbone = movinet.Movinet(model_id=config["model"]["model_id"])
  backbone.trainable = False

  # Set num_classes=600 to load the pre-trained weights from the original model
  model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=config["model"]["n_classes"])
  model.build([None, None, None, None, 3])

  # Load pre-trained weights
  wget.download("https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz", "./model_downloads")
  tar = tarfile.open("./model_downloads/movinet_a0_base.tar.gz")
  tar.extractall()
  tar.close()

  checkpoint_dir = f'movinet_{config["model"]["model_id"]}_base'
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()

  return backbone

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

def train_model(model, train_ds, val_ds, epochs, lr):
  """Trains the model using the training and validation datasets"""

  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

  results = model.fit(train_ds,
                      validation_data=val_ds,
                      epochs=epochs,
                      validation_freq=1,
                      verbose=1)
  
  return results

def get_actual_predicted_labels(dataset, model):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted