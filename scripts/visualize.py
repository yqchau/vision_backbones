import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms


def modify_state_dict(PATH):
    state_dict = torch.load(PATH)["state_dict"]
    # create new OrderedDict that does not contain `model.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[6:]  # remove `model.`
        new_state_dict[name] = v

    return new_state_dict


# model
model_name = "mobilenetv2_100"
pretrained = True
in_chans = 1
features_only = True
weights = "/home/ychau001/vision_backbones/logs/tune_32x32_onefourth/auto_tag_version_9/checkpoints/resnet_no_weighted.ckpt"
model = timm.create_model(
    model_name=model_name,
    pretrained=pretrained,
    in_chans=in_chans,
    features_only=features_only,
)

# state_dict = modify_state_dict(weights)
# state_dict.pop('fc.weight')
# state_dict.pop('fc.bias')
# model.load_state_dict(state_dict)
model.eval()

# data transformations
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
    ]
)
path = "/home/ychau001/driving_data/64x64_onefourth/test/right/06474.png"
inputs = Image.open(path).convert("RGB")
inputs = torch.unsqueeze(transform(inputs), dim=0)
outputs = model(inputs)

# visualize
nrows = 3
ncols = 2
i = 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 50))
axs[0][0].axis("off")
axs[0][0].imshow(inputs[0][0])
for output in outputs:
    output = output.detach().numpy()[0]
    # print(output.shape)
    avg = np.mean(np.mean(output, axis=1), axis=1)
    signal = np.argmax(avg)

    col = i % ncols
    row = i // ncols
    axs[row][col].axis("off")
    axs[row][col].imshow(output[signal])
    # axs[row][col].imshow(output[0].transpose(
    #     0, 2).sum(-1).transpose(0, 1).detach().numpy())
    i += 1

plt.show()
