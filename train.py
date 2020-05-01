import argparse

import fun_lib.py as fn

parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('--data_dir', nargs='*', action="store", default="./flowers/",
                    metavar='', help="Define the directory for data ")

parser.add_argument('--gpu', dest="gpu", action="store", default="gpu",
                    metavar='', help="GPU training")

parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001,
                    metavar='', help="Learning rate. Default = 0.001")

parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=10,
                    metavar='', help="Number of epochs. Default = 10")

parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str,
                    metavar='', help="CNN model architecture: vgg16")

args = parser.parse_args()


trainloader, validloader, testloader, cat_to_name, train_data = fn.load_data(args.data_dir)
device, model, criterion, opt = fn.build_model(args.gpu, args.arch)
fn.train_model(model, criterion, opt, trainloader, validloader, args.epochs, 3, args.gpu)
fn.test_model(model, testloader, device, criterion)
fn.save_checkpoint(args.epochs, 64, model, opt, criterion, train_data)

print("All  Done. The Model is trained")
