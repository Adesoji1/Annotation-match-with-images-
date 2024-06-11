import os
import json
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import jaccard_score

# Define paths (change these paths accordingly)
dataset_path = 'path/to/your/downloaded/osdar23_dataset'
output_model_path = 'path/to/output/model.pth'

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Custom dataset class
class OSDARDataset(TorchDataset):
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        annot = self.annotations[idx]
        if self.transform:
            image = self.transform(image)
        return image, annot

def load_annotations(folder_path):
    annotations = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    annotations.update(data)
    return annotations

def match_images_annotations(image_folder, annotations):
    matched_data = []
    for img_file in os.listdir(image_folder):
        if img_file.endswith('.png') or img_file.endswith('.jpg'):
            img_path = os.path.join(image_folder, img_file)
            image_id = os.path.splitext(img_file)[0]
            if image_id in annotations:
                matched_data.append((img_path, annotations[image_id]))
    return matched_data

def annotations_to_tensor(annotations):
    tensor_annotations = []
    for annot in annotations:
        bbox = annot['bbox']  # Assuming annotation has 'bbox' field
        tensor_annotations.append(torch.tensor(bbox, dtype=torch.float32))
    return tensor_annotations

def plot_bounding_boxes(image, bboxes):
    plt.imshow(image)
    ax = plt.gca()
    for bbox in bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, color='red')
        ax.add_patch(rect)
    plt.show()

def iou_score(pred_boxes, true_boxes):
    intersection = (min(pred_boxes[0] + pred_boxes[2], true_boxes[0] + true_boxes[2]) - max(pred_boxes[0], true_boxes[0])) * (min(pred_boxes[1] + pred_boxes[3], true_boxes[1] + true_boxes[3]) - max(pred_boxes[1], true_boxes[1]))
    union = (pred_boxes[2] * pred_boxes[3]) + (true_boxes[2] * true_boxes[3]) - intersection
    return intersection / union

def main():
    # Load annotations
    annotations = load_annotations(dataset_path)
    
    # Retrieve RGB annotations
    rgb_annotations = {k: v for k, v in annotations.items() if 'rgb' in k.lower()}
    
    # Match images with annotations
    image_folder = os.path.join(dataset_path, 'RGB')
    matched_data = match_images_annotations(image_folder, rgb_annotations)
    
    # Separate image paths and annotations
    image_paths, raw_annotations = zip(*matched_data)
    
    # Convert annotations to tensors
    tensor_annotations = annotations_to_tensor(raw_annotations)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    dataset = OSDARDataset(image_paths, tensor_annotations, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Load a simple VGG model
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 4)  # Adjust output layer
    model = model.to(device)
    
    # Training the model (simplified example)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in dataloader:
            images = images.to(device)
            targets = torch.stack(targets).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Save the model
    torch.save(model.state_dict(), output_model_path)
    
    # Evaluate the model
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = torch.stack(targets).to(device)
            outputs = model(images)
            for i in range(len(images)):
                iou = iou_score(outputs[i].cpu().numpy(), targets[i].cpu().numpy())
                iou_scores.append(iou)
                plot_bounding_boxes(images[i].cpu().permute(1, 2, 0).numpy(), [outputs[i].cpu().numpy()])
            break  # Plot only one batch for example
    
    # Print performance metric
    print(f'Average IoU score: {np.mean(iou_scores):.4f}')

if __name__ == "__main__":
    main()
