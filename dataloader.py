import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def cifar10_load_data(data_dir,batch_size,num_workers,valid_size=0.1,deterministic=False):
	"""Load the CIFAR10 dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
	transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    worker_init_fn = __deterministic_worker_init_fn if deterministic else None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    valid_loader = None
    if split > 0:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    testset = datasets.CIFAR10(root=data_dir, train=False,
                               download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    input_shape = __image_size(train_dataset)

    # If validation split was 0 we use the test set as the validation set
    return train_loader, valid_loader or test_loader, test_loader, input_shape