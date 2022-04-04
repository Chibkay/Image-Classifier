from collections import OrderedDict
import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms, datasets, models
import argparse


def load_data(data_dir):
    '''Load data'''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)
    return train_data, valid_data, test_data


def build_model(arch='vgg19', hidden_units=2960, lr=0.001):
    # function to define model, criterion and optimizer
    models_list = {'vgg19': models.vgg19(pretrained=True),
                   'vgg11': models.vgg11(pretrained=True),
                   'vgg13': models.vgg13(pretrained=True),
                   'vgg16': models.vgg16(pretrained=True)}

    model = models_list[arch]

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units,  bias=True)),
        ('Relu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102,  bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    return model, criterion, optimizer


def validation(model, testloader, criterion, device='cuda'):
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def train_model(model, train_data, valid_data, validation, epochs=8,
                device='cuda'):

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    running_loss = 0
    steps = 0
    print_every = 5

    # train the model
    for epoch in range(epochs):
        model.to(device)
        for inputs, labels in trainloader:
            steps += 1
            # move inputs and labels tensors to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)
            
                print(f"Epoch {epoch+1}/{epochs}.."
                 f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()


    model.class_to_idx = trainset.class_to_idx

    return model


def mk_checkpoint(model, arch='vgg19', hidden_units=2960):
    # create a checkpoint 
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }

    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('data', help='data directory')
    parser.add_argument('--save_dir', help='save directory')
    parser.add_argument(
        '--arch', choices=['vgg19', 'vgg11', 'vgg13', 'vgg16'],
        help='model architecture')
    parser.add_argument('--learning_rate',
                        help='the learning rate',  type=float)
    parser.add_argument('--epochs', help='number of epochs',  type=int)
    parser.add_argument('--hidden_units', help='hidden units',  type=int)
    parser.add_argument('--gpu', help='execution on gpu', action="store_true")
    args = parser.parse_args()

    arch = args.arch if args.arch else 'vgg19'
    hidden_units = args.hidden_units if args.hidden_units else 2960
    lr = args.learning_rate if args.learning_rate else 0.001
    epochs = args.epochs if args.epochs else 8
    device = 'cuda' if args.gpu else 'cpu'

    train_data, valid_data, test_data = load_data(args.data)
    model, criterion, optimizer = build_model(
        arch=arch, hidden_units=hidden_units, lr=lr)
    model = train_model(model, train_data, valid_data,
                        validation, epochs, device=device)
    checkpoint = mk_checkpoint(model, arch=arch, hidden_units=hidden_units)

    if args.save_dir:
        torch.save(checkpoint, args.save_dir + 'checkpoint_1.pth')
    else:
        torch.save(checkpoint, 'checkpoint_1.pth')