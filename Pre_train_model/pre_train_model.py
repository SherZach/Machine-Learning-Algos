import torchvision.models as models
import torch
from torchvision import datasets, transforms as T
from PIL import Image
import torch.nn.functional as F

alexnet = models.alexnet(pretrained=True)
googlenet = models.googlenet(pretrained=True)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])

image = Image.open("Radiator2.jpg")
image.show()
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
tens_img = transform(image)
# Add dimension to the tensor (making it 4 dimensions)
tens_img = tens_img.unsqueeze(0)

# Put model in test mode
alexnet.eval()
googlenet.eval()

# Get a list of each class name from the imagenet classes
with open('imagenet_classes.txt') as labels:
    classes = [i.strip() for i in labels.readlines()]

# Run model
output = alexnet(tens_img)
output2 = googlenet(tens_img)
# Convert output to a percentage chance of it being each class
percentage = F.softmax(output, dim=1)[0] * 100.0
percentage2 = F.softmax(output2, dim=1)[0] * 100.0

# get a list of tuples of each percentage and class name
results = [(percentage[i].item(), classes[i]) for i, x in enumerate(percentage)]
results.sort(reverse=True)
results2 = [(percentage2[i].item(), classes[i]) for i, x in enumerate(percentage2)]
results2.sort(reverse=True)

print("Alexnet results:\n\n", results[0:10])
print()
print("Googlenet results:\n\n", results2[0:10])

