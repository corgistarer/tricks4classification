import numpy as np
import torch

train_loader = None
model = None
criterion = None
accuracy = None
beta = 0.3  # hyperparameter
for (images, targets) in train_loader:
    I_x, I_y = images.size()[2:]

    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))

    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, I_y - h, I_y - h]

    # select and crop four images
    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        index = torch.randperm(images.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = targets[index].cuda()
        W_[k] = w_[k] * h_[k] / (I_x * I_y)

    # patch cropped images
    patched_images = torch.cat((torch.cat((cropped_images[0], cropped_images[1]), 2),
                                torch.cat((cropped_images[2], cropped_images[3]), 2)),
                               3)

    # patch_iamges = patch_images.cuda()

    # get output
    output = model(patched_images)

    # calculate loss and accuracy
    loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])
    acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
