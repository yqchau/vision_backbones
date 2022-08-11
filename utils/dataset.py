import os
import random
import shutil


def split(paths, n=40, seed=None):
    """given a list of paths, split them into 2 according to n."""
    if seed:
        random.seed(seed)

    length = len(paths)
    assert length > n, "n is greater than length of paths"

    random.shuffle(paths)
    extracted_paths = paths[:n]
    remaining_paths = paths[n:]
    assert len(extracted_paths) == n
    assert len(remaining_paths) + len(extracted_paths) == length

    return extracted_paths, remaining_paths


def move_files(paths, to):
    """
    paths: list of absolute paths of files to be moved
    to: absolute path of directory to be moved to
    """

    for path in paths:
        shutil.move(path, to)


def list_directory(dir, directory_only=False, files_only=False):
    """list the content of a directory when given the absolute path of the
    directory.

    It is recursive by default unless directory_only is set to True.
    """
    if directory_only and files_only:
        raise Exception("Choose either directory_only or files_only but not both")

    if directory_only:
        return [
            os.path.join(dir, path)
            for path in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, path))
        ]
    if files_only:
        return [
            os.path.join(dir, path)
            for path in os.listdir(dir)
            if not os.path.isdir(os.path.join(dir, path))
        ]
    return [os.path.join(dir, path) for path in os.listdir(dir)]


if __name__ == "__main__":
    dir = "/home/ml2/Desktop/sign_dataset"
    classes = list_directory(dir, directory_only=True)
    new_dir = "/home/ml2/Desktop/sign_dataset_cleaned"

    # create directory if does not exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        os.makedirs(os.path.join(new_dir, "train"))
        os.makedirs(os.path.join(new_dir, "val"))
        os.makedirs(os.path.join(new_dir, "test"))

    for object in classes:
        # loop each class
        object_name = object.split("/")[-1]
        paths = list_directory(object, files_only=True)

        # split into train, val and test
        test_set, train_set = split(paths, n=40, seed=10)
        val_set, train_set = split(train_set, n=40, seed=10)

        # create dir for files to be saved
        os.makedirs(os.path.join(new_dir, "train", object_name))
        os.makedirs(os.path.join(new_dir, "val", object_name))
        os.makedirs(os.path.join(new_dir, "test", object_name))

        # move files
        move_files(train_set, os.path.join(new_dir, "train", object_name))
        move_files(val_set, os.path.join(new_dir, "val", object_name))
        move_files(test_set, os.path.join(new_dir, "test", object_name))

        print(f"{object_name} done..")

    print("completed")
