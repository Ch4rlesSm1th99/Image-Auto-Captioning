import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance, ImageOps
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


image_dir = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\flickr30k\images"
captions_path = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\flickr30k\captions.csv"
output_dir = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k"


def load_data():
    captions = pd.read_csv(captions_path, sep='|')
    captions.columns = captions.columns.str.strip()
    image_caption_pairs = captions[['image_name', 'comment']]
    return image_caption_pairs

# augmentations are relatively weak
def augment_image(image_path):
    image = Image.open(image_path).convert('RGB')
    augmentations = []
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    ])
    for _ in range(5):
        augmented_image = transform(image)
        augmentations.append(augmented_image)
    return augmentations



def save_image(image, path):
    image.save(path)


# applies augmentation and creates dataset
def create_dataset():
    image_caption_pairs = load_data()
    images = list(set(image_caption_pairs['image_name']))

    # split into train val test sets
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    splits = {'train': train_images, 'val': val_images, 'test': test_images}

    for split_name, split_images in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        dataset = []
        for image_name in tqdm(split_images, desc=f"Processing {split_name} set"):
            image_path = os.path.join(image_dir, image_name)
            captions = image_caption_pairs[image_caption_pairs['image_name'] == image_name]['comment'].values

            # Perform 5 augmentations for each image
            augmented_images = augment_image(image_path)
            for i, aug_image in enumerate(augmented_images):
                caption = captions[i % len(captions)]  # assign one of the five captions to aug images
                save_path = os.path.join(split_dir, f"{os.path.splitext(image_name)[0]}_aug_{i}.jpg")
                save_image(aug_image, save_path)
                dataset.append({'image': save_path, 'caption': caption, 'filename': image_name,
                                'image_id': f"{image_name}_aug_{i}"})


        torch.save(dataset, os.path.join(output_dir, f"{split_name}_dataset.pt"))


if __name__ == "__main__":
    create_dataset()
