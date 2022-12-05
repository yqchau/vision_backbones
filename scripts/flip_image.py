import argparse
import glob
import os
import shutil

import cv2


def flip_image(input_dir, src_path, dst_path):

    output_dir = input_dir + f"/{dst_path}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    paths = sorted(glob.glob(input_dir + f"/{src_path}/*.png"))
    all_paths = sorted(glob.glob(input_dir + "/*/*.png"))
    # print(all_paths)
    idx = len(all_paths)

    for path in paths:
        new_path = path.replace(src_path, dst_path)
        img = cv2.imread(path)
        flipped_img = cv2.flip(img, 1)

        arr = new_path.split("/")
        arr[-1] = str(idx).zfill(5) + ".png"
        new_path = "/".join(arr)
        print(new_path)

        cv2.imwrite(new_path, flipped_img)
        idx += 1


def move_files(src_dir, dst_dir):

    for file_name in os.listdir(src_dir):
        # construct full file path
        source = src_dir + "/" + file_name
        destination = dst_dir + "/" + file_name
        # move only files

        if os.path.isfile(source):
            shutil.move(source, destination)
            print("Moved:", file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", help="Directory of Training Data to perform Flip"
    )
    args = parser.parse_args()

    input_dir = args.input_dir

    flip_image(input_dir, "left", "flipped_to_right")
    flip_image(input_dir, "right", "flipped_to_left")

    move_files(input_dir + "/flipped_to_right", input_dir + "/right")
    move_files(input_dir + "/flipped_to_left", input_dir + "/left")

    os.removedirs(input_dir + "/flipped_to_right")
    os.removedirs(input_dir + "/flipped_to_left")
