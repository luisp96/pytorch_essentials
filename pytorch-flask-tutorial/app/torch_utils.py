import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image # converts image to the tensor

# load model
# image -> tensor, make sure to apply the same transformations as the original model so transforms.ToTensor() and ransforms.Normalize((0.1307,), (0.3081,))
# predict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#LOAD MODEL
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
# saved model path
PATH = "mnist_ffn.pth"
# instantiate model, load dict from saved model(while loading it into the GPU) and set the model to eval mode
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

# IMAGE -> TENSOR
# the original transform were just to tensor and normalization, but we need two more transforms here:
# 1.- We want to use only use the grayscale(single channel) for the images, instead of the regular 3 RGB channel
# 2.- We need to ensure that the size is 28x28, because our input size is 784(which is 28x28)
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), # we only want one output channel for grayscale
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])  # mean and std

    # we want to create a PILImage from the image bytes
    image = Image.open(io.BytesIO(image_bytes))
    # returns a new tensor and inserts a dimension of 1 at position 0
    # (this is because during training we had the first dimension as the batch size which we don't need here since we are only doing one image at a time)
    return transform(image).unsqueeze(0)

# PREDICT
def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28 * 28).to(device)
    outputs = model(images)
    # print(f'Output shape: {outputs.size()}, Output Raw: {outputs}, Outputs.Data: {outputs.data}')
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1) # max returns a tuple (value, index) of the highest number in the tensor. Which the index is the position in our class/label array if you will
    # print(f'Predicted label: {predicted}, Predicted label as an item: {predicted.item()}')
    return predicted