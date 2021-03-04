from torch.utils.data import DataLoader


def create_dataset(name, data_root, **kwargs):
    # datasets for image restoration
    if name == 'gopro':
        from data.gopro import GoPro as D
    elif name == 'reds':
        from data.reds120 import REDS120fps as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(data_root, **kwargs)

    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           name))
    return dataset
