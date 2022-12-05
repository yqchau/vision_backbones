import os
import tarfile

dir = "/home/ychau001/Downloads/ImageNet1k"
os.mkdir(dir + "/train")

i = 0
for path in os.listdir(f"{dir}/train_zip"):

    actual = path.replace(".tar", "")
    os.mkdir(f"/home/ychau001/Downloads/ImageNet1k/train/{actual}")

    tar = tarfile.open(dir + "/train_zip/" + path)
    tar.extractall(f"/home/ychau001/Downloads/ImageNet1k/train/{actual}")
    print(i)

    i += 1
