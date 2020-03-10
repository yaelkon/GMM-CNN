import os
import json


def makedir(dir_path):
    """Make directory in specified path.
    Args:
        dir_path (str): directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_to_file(file_dir, objs_name, objs):
    """Save objects to file
        Args:
        file_dir: path containing the source files.
        obj_names: list containing the objects' name.
        obj: list of objects that will be saved.
    """
    for obj, name in zip(objs, objs_name):
        JS = os.path.join(file_dir, name + '.json')
        with open(JS, 'w') as f:
            json.dump(obj, f)


def load_from_file(file_dir, obj_name):
    """Load JSON objects from file
        Args:
        file_dir: path containing the source files.
        obj_names: list containing the objects' name.

    """
    data = []
    for name in obj_name:
        JS = os.path.join(file_dir, name + '.json')
        with open(JS) as json_file:
            data.append(json.load(json_file))

    return data