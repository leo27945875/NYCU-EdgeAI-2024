from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import os
import torch
from pippy.IR import *
from torch.profiler import profile, ProfilerActivity


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def build_dataset_CIFAR100(is_train, data_path):
    transform = build_transform(is_train)
    dataset = datasets.CIFAR100(data_path, train=is_train, transform=transform, download=True)
    nb_classes = 100
    return dataset, nb_classes


def build_dataset_CIFAR10(is_train, data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    dataset = datasets.CIFAR10(data_path, train=is_train, transform=transform, download=True)
    nb_classes = 10

    return dataset, nb_classes


def build_transform(is_train):
    input_size = 224
    eval_crop_ratio = 1.0

    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.0,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(input_size / eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def prepare_data(batch_size, data='cifar-100'):

    if data == 'cifar-100':
        train_set, nb_classes = build_dataset_CIFAR100(is_train=True, data_path='./data/cifar100')
        test_set, _ = build_dataset_CIFAR100(is_train=False, data_path='./data/cifar100')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    elif data == 'cifar-10':
        train_set, nb_classes = build_dataset_CIFAR10(is_train=True, data_path='./data/cifar10')
        test_set, _ = build_dataset_CIFAR10(is_train=False, data_path='./data/cifar10')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    else:
        raise NotImplementedError

    return train_loader, test_loader, nb_classes


def evaluate_output(output, labels):
    correct = 0
    total = 0

    _, predicted = torch.max(output, 1)
    total += labels.size(0)

    correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy


def getMiniTestDataset(nimages):
    # Create a test_loader with batch size = 1
    _, test_loader, _ = prepare_data(batch_size=1)

    # Prepare to collect 5 images per class
    class_images = [[] for _ in range(100)]
    images_per_class = nimages // 100

    # Iterate through the data
    for (image, label) in test_loader:
        if len(class_images[label]) < images_per_class:
            class_images[label].append((image, label))
        if all(len(images) == images_per_class for images in class_images):
            break  # Stop once we have 10 images per class

    # flatten class_images
    mini_test_dataset = []
    for images in class_images:
        mini_test_dataset.extend(images)

    
    images, labels = zip(*mini_test_dataset)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return (images, labels)

# TODO: Run the stage of each rank and return the output from last rank
def run_stage(stage, rank, world_size, imgs):
    output = None
    if rank == 0:
        stage(imgs)
    elif rank == world_size - 1:
        output = stage()
    else:
        stage()

    print(f"Rank {rank} completes")
    return output

# TODO: Run the stage of each rank with profiler enable
def run_stage_with_profiler(stage, rank, world_size, imgs):
    with profile(
        activities=[ProfilerActivity.CPU], 
        profile_memory=True, 
        record_shapes=True,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/rank_{rank}/')
    ) as prof:
        output = run_stage(stage, rank, world_size, imgs)
    
    prof.export_chrome_trace(f"{os.path.splitext(os.path.basename(__file__))[0]}_{rank=}.json")
    return output

def run_serial(model, imgs):

    result = None

    # for i in tqdm(range(num_iter)):
    for img in tqdm(imgs):
        
        if result == None:
            output = model(img)
            result = output
        else:
            output = model(img)
            result = torch.cat((result, output), dim=0)

    return result

# TODO: Run model with profiler enable
def run_serial_with_profiler(model, imgs):
    with profile(
        activities=[ProfilerActivity.CPU], 
        profile_memory=True, 
        record_shapes=True,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/serial/')
    ) as prof:
        output = run_serial(model, imgs)
    
    prof.export_chrome_trace(f"{os.path.splitext(os.path.basename(__file__))[0]}_serial.json")
    return output