import torch
from PIL import Image
from torch.autograd import Variable


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    # batch = torch.div(batch, 255.0)
    out =batch - Variable(mean)
    out = out / Variable(std)
    return out

def quantize(img, rgb_range=1):
    # change to 0~255
    if isinstance(img, torch.Tensor):
        img = img.clamp(0, 1)
        img = img.mul(255/rgb_range).round()
    
    elif isinstance(img, np.ndarray):
        img = img.clip(0, 1)
        img = np.round(img*255)
    
    else:
        raise NotImplementedError()

    return img
    

def meanshift(batch, rgb_mean, rgb_std, device, norm=True):
    if isinstance(rgb_mean, list):
        rgb_mean = torch.Tensor(rgb_mean)
    if isinstance(rgb_std, list):
        rgb_std = torch.Tensor(rgb_std)

    rgb_mean = rgb_mean.reshape(1,3,1,1).to(device)
    rgb_std = rgb_std.reshape(1,3,1,1).to(device)

    if norm:
        return (batch - rgb_mean) / rgb_std
    else:
        return (batch * rgb_std) + rgb_mean
