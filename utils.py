import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime


# fs ##########################################################################
def path_is_file(file_path):
    """
    Check if a file exists at the given path.
    :param file_path: The path to the file
    :return: bool, True if the file exists, False otherwise
    """
    return os.path.isfile(file_path)


def path_join(directory, filename):
    return os.path.join(directory, filename)


def path_dirname(file_path):
    """
    Get the directory name from a given file path.
    :param file_path: The path to the file
    :return: The directory name
    """
    dirname = os.path.dirname(file_path)
    return dirname


def path_basename(path):
    """
    Get the basename (last component) of a file path.
    :param path: File path as a string
    :return: Basename of the path
    """
    return os.path.basename(path)


def mkdir(directory):
    os.makedirs(directory, exist_ok=True)


# log #########################################################################
def log_init(log_file):
    """
    Create or overwrite a log file.
    :param log_file: Path to the log file
    """
    try:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(log_file, 'w') as f:
            f.write(f"Log created: {date}\n")
        print(f"\tlog file '{log_file}' created/overwritten successfully.")
    except Exception as e:
        print(f"utils:log_init: error creating file '{log_file}': {e}")
        exit(1)


def log(message, log_file, silent=False):
    """
    Helper function to log messages to both a file and optionally to stdout.
    :param message: Message to log
    :param log_file: Path to the log file
    :param silent: If True, suppresses stdout
    """
    try:
        with open(log_file, 'a') as f:  # NOTE: no need to close after this
            f.write(message + '\n')
    except Exception as e:
        print(f"utils:log: error writing message to '{log_file}': {e}")
        exit(1)
    if not silent:
        print(message)


# fig #########################################################################
def fig_save(fig, path):
    """
    Helper function to save a plot.
    :param fig: The figure to save
    :param path: The file path where the figure should be saved
    """
    try:
        fig.savefig(path)
        print(f"\tsaved figure to '{path}'.")
    except Exception as e:
        print(f"fig_save: Error saving figure to '{path}': {e}")
        exit(1)
    finally:
        plt.close(fig)


# csv #########################################################################
def csv_load(path):
    """
    Load csv from path.
    :param path: csv path
    :return: DataFrame|None
    """
    try:
        df = pd.read_csv(path)
        print(f"\tloaded file from {path}.")
    except FileNotFoundError:
        print(f"utils:csv_load: The file '{path}' was not found.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"utils:csv_load: The file '{path}' is empty.")
        exit(1)
    except pd.errors.ParserError:
        print(f"utils:csv_load: There was a problem parsing the file '{path}'.")
        exit(1)
    except Exception as e:
        print(f"utils:csv_load: An error occurred loading file '{path}': {e}")
        exit(1)
    return df


def csv_save(data, path):
    """
    Parse the DataFrame.
    :param data:  DataFrame
    :param path:  String
    """
    try:
        data.to_csv(path, index=False)
        print(f"\tdata saved successfully to '{path}'.")
    except Exception as e:
        print(f"utils:csv_save: error saving data to '{path}': {e}")
        exit(1)


def load_data(path):
    """
    Transform data in X train and True labels data-sets
    :param data:  Dataframe
    :return:  X train and true labels data-sets
    """
    data = csv_load(path)
    y = data.iloc[:, 0].to_numpy()  # labels
    X = data.iloc[:, 1:].to_numpy()  # data
    return X, y


# json ########################################################################
def json_load(path):
    """
    Load a JSON file and return the corresponding Python object.
    :param path: Path to the JSON file.
    :return: Python object (dict, list, str, int, etc.)|None
    """
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
        print(f"\tLoaded object from '{path}'.")
    except FileNotFoundError:
        print(f"utils:json_load: The file '{path}' was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"utils:json_load: There was a problem decoding the JSON file '{
              path}'.")
        exit(1)
    except Exception as e:
        print(f"utils:json_load: An error occurred loading object from '{
              path}': {e}")
        exit(1)
    return obj


def json_save(obj, path, silent=False):
    """
    Save a JSON-serializable object to a file.
    :param obj: JSON-serializable object (dict, list, str, int, etc.)
    :param path: Path to save the JSON file.
    """
    try:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=4)
        if not silent:
            print(f"\tObject saved successfully to '{path}'.")
    except TypeError as e:
        print(f"utils:json_save: error serializing object: {e}")
        exit(1)
    except Exception as e:
        print(f"utils:json_save: error saving object to '{path}': {e}")
        exit(1)


def transformar_datos(data):
    """
    Transform data in X train and True labels data-sets
    :param data:  Dataframe
    :return:  X train and true labels data-sets
    """

    data = pd.DataFrame(data)
    # Seleccionar la primera columna
    primera_columna = data.iloc[:, 0].astype(int)

    # Convertir la columna en un np.array
    true_labels = np.array(primera_columna)

    train_dim = pd.DataFrame(data.loc[:, data.columns != 'PU'])

    for columna in train_dim.columns:
        train_dim[columna] = train_dim[columna].astype(float)

    X = train_dim.values.astype(float)

    return X, true_labels


# model #######################################################################
def save_model(model, model_path, silent=False):
    """
    Save created model
    :param model
    :param model_path
    """
    joblib.dump(model, model_path)
    if not silent:
        print(f"Best model saved in '{model_path}'")


def load_model(model_path):
    """
    Load model
    :param model_path
    :return:  model
    """
    return joblib.load(model_path)

