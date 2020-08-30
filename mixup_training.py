import numpy as np
import torch

train_loader = None
mixup_alpha = 0.5
model = None
criterion = None
accuracy = None
for (images, labels) in train_loader:
    l = np.random.beta(mixup_alpha, mixup_alpha)
    index = torch.randperm(images.size(0))
    images_a, images_b = images, images[index]
    labels_a, labels_b = labels, labels[index]

    mixup_images = l * images + (1 - l) * images_b

    outputs = model(mixup_images)
    loss = l * criterion(outputs, labels_a) + (1 - l) * criterion(outputs, labels_b)
    acc = l * accuracy(outputs, labels_a)[0] + (1 - l) * accuracy(outputs, labels_b)[0]
