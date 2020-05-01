import torch
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import torch.utils.data


# Train
def load_data(data_dir="./flowers"):
    """
    Receives the location of the image files,
    applies the necessary transformations,
    converts the images to tensor in order to be able to be fed into the neural network
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    # label mapping
    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)

    print("================- Done || Loading || Data -================")

    return trainloader, validloader, testloader, cat_to_name, train_data


def load_classifier():
    """
    :return: vgg16 classifier parameters
    """
    input_size = 25088
    hidden_sizes = 1024
    output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(input_size, hidden_sizes)),
        ("relu1", nn.ReLU()),
        ("dropout1", nn.Dropout(0.5)),
        ("fc2", nn.Linear(hidden_sizes, 120)),
        ("relu2", nn.ReLU()),
        ("dropout2", nn.Dropout(0.2)),
        ("fc3", nn.Linear(120, 90)),
        ("relu3", nn.ReLU()),
        ("fc4", nn.Linear(90, 80)),
        ("relu4", nn.ReLU()),
        ("fc5", nn.Linear(80, output_size)),
        ("output", nn.LogSoftmax(dim=1))
    ]))
    return classifier


def build_model(power, arch):
    # use gpu model
    device = torch.device("cuda" if (torch.cuda.is_available() and power.gpu == "gpu") else "cpu")

    # define model
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        print("Error: No model architecture defined!")

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Specify classifier
    model.classifier = load_classifier()

    # specify loss function
    criterion = nn.NLLLoss()

    # specify optimizer
    opt = optim.Adam(model.classifier.parameters(), lr=0.001)

    # move the model to device
    model.to(device)

    print("================- Done || Model || Building -================")
    return device, model, criterion, opt


def train_model(model, criterion, optimizer, trainloader, validloader, epochs=3, print_every=3, power='gpu'):
    """
    trains the model over a certain number of epochs,
    display the training, validation and accuracy
    """
    steps = 0
    running_loss = 0

    print("================- Training || Start -================")
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            if torch.cuda.is_available() and power == "gpu":
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward process
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print process
            if steps % print_every == 0:
                model.evel()
                validloss = 0
                accuracy = 0

                for v_inputs, v_labels in validloader:
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        v_inputs, v_labels = v_inputs.to('cuda'), v_labels.to('cuda')
                        model.to('cuda')

                        with torch.no_grad():
                            outputs = model.forward(v_inputs)
                            validloss = criterion(outputs, v_labels)

                            ps = torch.exp(outputs)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += equals.type_as(torch.FloatTensor()).mean()

                        print(f"Epoch {epochs + 1}/{epochs}.. "
                              f"Train loss: {running_loss / print_every:.3f}.. "
                              f"valid loss: {validloss / len(validloader):.3f}.. "
                              f"valid accuracy: {accuracy / len(validloader):.3f}")

                        running_loss = 0
    print("================- Training || Done -================")


def test_model(model, testloader, device, criterion):
    print("================- Training || Start -================")

    test_loss = 0
    accuracy = 0

    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)

            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Test loss: {test_loss / len(testloader):.3f} "
                  f"Test accuracy: {accuracy / len(testloader):.3f} ")

    print('Accuracy of the network on the test dataset: %d %%' % (100 * accuracy / len(testloader)))
    print("================- Testing || Done -================")


def save_checkpoint(epochs, batch_size, model, optimizer, criterion, train_dataset):
    """
    save trained model
    """
    checkpoint = {"model": models.vgg16(pretrained=True),
                  "input_size": 2208,
                  "output_size": 102,
                  "epochs": epochs,
                  "batch_size": batch_size,
                  "state_dict": model.state_dict(),
                  "state_features_dict": model.features.state_dict(),
                  "state_classifier_dict": model.classifier.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "criterion_state_dict": criterion.state_dict(),
                  "class_to_idx": train_dataset.class_to_idx,
                  }
    torch.save(checkpoint, 'checkpoint.pth')

    model.cpu()

    load_model = torch.load('checkpoint.pth')
    load_model.keys()

    print("================- Done || Saving || Model -================")


# Predict
def load_checkpoint(pth_dir='checkpoint.pth'):
    """
    :param pth_dir: checkpoint directory
    :return: nn model prediction
    """
    device, model, criterion, opt = build_model('cpu', 'vgg16')
    check = torch.load(pth_dir, map_location='cpu')
    model.load_state_dict(check["state_dict"], strict=False)
    print("================- Done || Rebuild || Model -================")
    return model


def process_image(img_path):
    """
    :param img_path: The directory for image
    :return: Tensor image
    """
    img_path = str(img_path)
    img = Image.open(img_path)
    trans = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
    tensor_img = trans(img)
    return tensor_img


def predict_class(cat_to_name, img_path, top_k=5, device="gpu"):
    """
     Predict the class of the image
    :param cat_to_name: flowers name
    :param img_path: The path to the image
    :param top_k: The numbers of predictions
    :param device: Use cpu vs gpu if available
    :return: top_k probability
    """

    model = load_checkpoint()
    model.eval()

    img = process_image(img_path)
    img = img.unsqueeze(0)

    with torch.no_grad():
        if model == 0:
            print("LoadCheckpoint: ERROR - Checkpoint load failed")
        else:
            print("LoadCheckpoint: Checkpoint loaded")

        if torch.cuda.is_available() and device == "gpu":
            model.to('cuda')
        else:
            model.to('cpu')
            device = 'cpu'

        inputs = img.to(device, dtype=torch.float)
        log_ps = model.forward(inputs)
        ps = torch.exp(log_ps)
        classes = ps.topk(top_k, dim=1)
    model.train()

    classes_ps = classes[0]
    classes_ps = classes_ps.cpu().tolist()
    classes_ps = [item for sublist in classes_ps for item in sublist]

    # Extract predicted class index, copy tensor to CPU, convert to list.
    classes_idx = classes[1]
    classes_idx = classes_idx.cpu().tolist()

    # Get predicted flower names from cat_to_name
    class_names = [cat_to_name.get(str(idx)) for idx in np.nditer(classes_idx)]

    print("Class Index: ", classes_idx)
    print("Class Names: ", class_names)
    print("Class Probabilities: ", classes_ps)

    return classes_ps, class_names, ps, classes


"""
def predict(img_path, top_k=5):
    load_checkpoint("checkpoint.pth")
    img = process_image(img_path)
    predict_class(img_path, top_k)
    print("================- -:) Done (:- -================")
"""
