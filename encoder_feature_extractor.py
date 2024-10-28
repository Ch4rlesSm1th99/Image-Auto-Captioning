import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Paths
dataset_paths = {
    'train': {
        'images': r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\train",
        'metadata': r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\train_dataset.pt"
    },
    'val': {
        'images': r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\val",
        'metadata': r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\val_dataset.pt"
    },
    'test': {
        'images': r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\test",
        'metadata': r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\test_dataset.pt"
    }
}

save_dir = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\flickr30k features"

# load models
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer
resnet.eval()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# prep for resnet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def load_dataset(split_name):
    dataset_path = dataset_paths[split_name]['metadata']
    return torch.load(dataset_path)


def extract_features():
    splits = ['train', 'val', 'test']

    for split in splits:
        dataset = load_dataset(split)
        image_dir = dataset_paths[split]['images']
        image_features = []
        caption_features = []

        for data in tqdm(dataset, desc=f"Extracting features for {split} set"):
            # image feature extraction
            image_path = os.path.join(image_dir, os.path.basename(data['image']))
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}. Skipping.")    # image in train not stored properly
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping.")
                continue

            image = image_transform(image).unsqueeze(0)  # add dimension for batch
            with torch.no_grad():
                image_feature = resnet(image).squeeze().numpy()
            image_features.append(image_feature)

            # caption feature extraction
            caption = data['caption']
            if not isinstance(caption, str):
                print(f"Warning: Caption is not a string for image {image_path}. Skipping.")
                continue

            inputs = bert_tokenizer(caption, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                caption_feature = bert_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
            caption_features.append(caption_feature)

        features = {'image_features': image_features, 'caption_features': caption_features}
        save_path = os.path.join(save_dir, f"{split}_extracted_features.pt")
        torch.save(features, save_path)


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)
    extract_features()
