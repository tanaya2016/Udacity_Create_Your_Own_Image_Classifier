#Imports here
import argparse
import json
import PIL
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms



# Options to run
# python predict.py flowers/test/100/image_07902.jpg model_checkpoint.pth --gpu
# python predict.py flowers/test/100/image_07902.jpg model_checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/100/image_07902.jpg')
    parser.add_argument('checkpoint_loc', metavar='checkpoint_loc', type=str, default='model_checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    check_model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    check_model.input_size = checkpoint['input_size']
    check_model.output_size = checkpoint['output_size']
    check_model.learning_rate = checkpoint['learning_rate']
    check_model.hidden_units = checkpoint['hidden_units']
    check_model.learning_rate = checkpoint['learning_rate']
    check_model.classifier = checkpoint['classifier']
    check_model.epochs = checkpoint['epochs']
    check_model.load_state_dict(checkpoint['state_dict'])
    check_model.class_to_idx = checkpoint['class_to_idx']
    check_model.optimizer = checkpoint['optimizer']
    return check_model


def process_image(image):
    image_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    pil_image = Image.open(image)
    tensor_image = image_transforms(pil_image)
    return tensor_image


def predict(image_path, model, top_k, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()

    with torch.no_grad():
        output = model.forward(image.cuda())

    probability = F.softmax(output.data, dim=1)

    top_prob = np.array(probability.topk(top_k)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(top_k)[1][0])]

    return top_prob, top_classes, device



def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint_loc = args.checkpoint_loc
    top_k = args.top_k
    category_names = args.category_names
    
    with open(args.category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    gpu = args.gpu

    model = load_checkpoint(checkpoint_loc)

    top_prob, classes, device = predict(image_path, model, top_k, gpu)

    

    labels = [cat_to_name[str(index)] for index in classes]

    print(f"Image Details: {image_path}")
    #print(labels)
    #print(probability)
    print(f"Results for Chosen Image are:\n")

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], top_prob[i]))


if __name__ == "__main__":
    main()