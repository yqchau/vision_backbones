import glob
import os

dir = "/home/ychau001/Downloads/ImageNet1k/train"
gt = "/home/ychau001/Downloads/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
val_all = "/home/ychau001/Downloads/ImageNet1k/val_all"

all_paths = glob.glob(val_all + "/*.JPEG")

if not os.path.exists(dir + "/../val"):
    os.mkdir(dir + "/../val")

folders = os.listdir(dir)
folders.sort()

y = list(map(lambda x: len(os.listdir(dir + f"/{x}")), folders))
print(min(y))
# f = open(gt)
# lines = f.readlines()
# print(lines)

# lines = list(map(lambda x: folders[int(x.replace('\n', '')) - 1], lines))

# print(lines[490])

# idx = 1
# for line in lines:
#     val = f'/home/ychau001/Downloads/ImageNet1k/val/{line}'
#     if not os.path.exists(val):
#         os.mkdir(val)

#     i = idx - 1

#     shutil.move(all_paths[i], val)
#     print(line)

#     idx += 1
