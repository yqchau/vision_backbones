import torch
import torchvision.datasets as datasets
from torchvision import transforms


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


if __name__ == "__main__":

    transform = transforms.Compose(
        [
            # you can add other transformations in this list
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dir = "/home/ml2/tl_good_ori/val"
    dataset_train = datasets.ImageFolder(train_dir, transform=transform)
    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(
        dataset_train.imgs, len(dataset_train.classes)
    )
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=512,
        # shuffle=True,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
    )

    for step, (x, y) in enumerate(train_loader):
        count = torch.bincount(y)
        print(count)
