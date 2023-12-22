from src.visualize.visualize import plot_confusion_matrix
from src.preprocessing.build_features import train_test_val_split
import src.datasets.load_datasets as ld
import src.models.train_models as tm
import json
import sys

def main(config):
    configM = config["model"]

    # Set up and load data
    subset_paths = ld.load_dataset(config["data"]["path"], config["data"]["dir"])

    train_ds, test_ds, val_ds =  train_test_val_split(subset_paths)

    fg = ld.FrameGenerator(subset_paths['train'], config["data"]["num_frames"], training = True)
    label_names = list(fg.class_ids_for_name.keys())

    # Load pre-trained model
    backbone = tm.load_movinet(config)

    # Train the model
    model = tm.build_classifier(configM["batch_size"], configM["num_frames"], configM["resolution"], backbone, 10)
    tm.train_model(model, train_ds, val_ds, configM["n_epochs"], configM["lr"])
    model.evaluate(test_ds, return_dict=True)

    # Calculate accuracy and plot
    actual, predicted = tm.get_actual_predicted_labels(test_ds, model)
    plot_confusion_matrix(actual, predicted, label_names, 'test')

if __name__ == "__main__":
    # Read and open config file specificed in argument
    args = sys.argv[1:]
    config_name = "config/" + args[0]
    with open(config_name, "r") as jsonfile:
        config = json.load(jsonfile)
    main(config)