import glob

from torchvision.io import read_image

mean = [0.0, 0.0, 0.0]
std = [0.0, 0.0, 0.0]

new_dir = "/home/ml2/sign_dataset/*/*.jpg"
paths = glob.glob(new_dir)

for path in paths:
    img = read_image(path).float()
    img /= 255
    img1, img2, img3 = img

    mean[0] += img1.mean().item()
    mean[1] += img2.mean().item()
    mean[2] += img3.mean().item()

    std[0] += img1.std().item()
    std[1] += img2.std().item()
    std[2] += img3.std().item()

    print(f"{path} done..")

mean = [i / len(paths) for i in mean]
std = [i / len(paths) for i in std]

print()
print(f"Mean: {mean}")
print(f"Std: {std}")
print("completed")
