import torchvision.models as models
import torch
from torchvision import datasets, transforms as T
from PIL import Image
import torch.nn.functional as F

model_list = [
    ("Alexnet", models.alexnet(pretrained=True)), 
    ("Googlenet", models.googlenet(pretrained=True)),
    ("Resnet18", models.resnet18(pretrained=True)),
    ("Densenet161", models.densenet161(pretrained=True)),
    ("Inception", models.inception_v3(pretrained=True)),
    ("Resnext50_32x4d", models.resnext50_32x4d(pretrained=True)),
    ("Wide_resnet50_2", models.wide_resnet50_2(pretrained=True)),
    ("Mnasnet1_0", models.mnasnet1_0(pretrained=True)),
    ("Shufflenet", models.shufflenet_v2_x1_0(pretrained=True))
]

for model in model_list:    
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    image = Image.open("Pre_train_model\dog1.jpg")
    # image.show()
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    tens_img = transform(image)
    # Add dimension to the tensor (making it 4 dimensions)
    tens_img = tens_img.unsqueeze(0)

    # Put model in test mode
    model[1].eval()

    # Get a list of each class name from the imagenet classes
    with open('Pre_train_model\imagenet_classes.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]

    # Run model
    output = model[1](tens_img)
    # Convert output to a percentage chance of it being each class
    percentage = F.softmax(output, dim=1)[0] * 100.0

    # get a list of tuples of each percentage and class name
    results = [(percentage[i].item(), classes[i]) for i, x in enumerate(percentage)]
    results.sort(reverse=True)

    print(model[0], "results:\n\n", results[0:10])
    print()

