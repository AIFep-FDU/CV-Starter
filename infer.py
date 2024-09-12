import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import json
from main import design_model
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = './checkpoints/best_model.pth'
class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']  
num_classes = len(class_names)

model = design_model(num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

resolution = 32
infer_transforms = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = infer_transforms(img)
    img = img.unsqueeze(0) 
    return img.to(device)

def infer(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img)
        prediction = output.argmax(dim=1, keepdim=True).item()
        return prediction

def save_inference_results(image_dir, output_json):
    results = {}
    for image_file in tqdm(sorted(os.listdir(image_dir))):
        base_name, ext = os.path.splitext(image_file)
        if ext in ('.jpg', '.jpeg', '.png'):
            image_path = os.path.join(image_dir, image_file)
            predicted_class_id = infer(image_path)
            predicted_class_name = class_names[predicted_class_id]
            results[base_name] = {
                "label": predicted_class_name,
                "label_id": predicted_class_id
            }

    with open(output_json, 'w') as json_file:
        json.dump(results, json_file, indent=4)


image_directory = './voc2007/test_1k'  
output_json_path = 'submission.json'
save_inference_results(image_directory, output_json_path)
print(f'Results saved to {output_json_path}')