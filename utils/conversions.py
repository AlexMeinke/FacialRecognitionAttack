import io
import torch
import torchvision.transforms
from PIL import Image


def PIL_to_binary(img):
    output = io.BytesIO()
    img.save(output, format="JPEG")
    hex_data = output.getvalue()

    return hex_data


def torch_to_PIL(tensor):
    temp = tensor.permute(1, 2, 0).detach().cpu()
    img = Image.fromarray((255.0 * temp).type(torch.uint8).numpy())
    return img


def PIL_to_torch(img):
    tensor = torchvision.transforms.ToTensor()(img)
    return tensor


def torch_to_binary(tensor):
    return PIL_to_binary(torch_to_PIL(tensor))
