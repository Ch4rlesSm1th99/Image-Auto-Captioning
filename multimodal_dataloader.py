import torch
from torch.utils.data import Dataset, DataLoader
import os


class MultimodalDataset(Dataset):
    def __init__(self, features_path, metadata_path, include_metadata=False):
        # load files with extracted features
        data = torch.load(features_path)
        self.image_features = data['image_features']
        self.caption_features = data['caption_features']

        # load metadata ie captions from when the dataset was built
        self.metadata = torch.load(metadata_path)
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        item = {
            'image_feature': torch.tensor(self.image_features[idx], dtype=torch.float),
            'caption_feature': torch.tensor(self.caption_features[idx], dtype=torch.float)
        }

        if self.include_metadata:
            metadata = self.metadata[idx]
            item['caption'] = metadata['caption']  # original captions not the features
            item['filename'] = metadata['filename']
            item['image_id'] = metadata['image_id']

        return item


def create_dataloaders(batch_size, train_features_path, val_features_path, train_metadata_path, val_metadata_path,
                       include_metadata=False):
    train_dataset = MultimodalDataset(train_features_path, train_metadata_path, include_metadata=include_metadata)
    val_dataset = MultimodalDataset(val_features_path, val_metadata_path, include_metadata=include_metadata)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


# batch inspection if interested
if __name__ == "__main__":
    train_features_path = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\flickr30k features\train_extracted_features.pt"
    val_features_path = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\flickr30k features\val_extracted_features.pt"
    train_metadata_path = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\train_dataset.pt"
    val_metadata_path = r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\val_dataset.pt"

    train_loader, val_loader = create_dataloaders(batch_size=32,
                                                  train_features_path=train_features_path,
                                                  val_features_path=val_features_path,
                                                  train_metadata_path=train_metadata_path,
                                                  val_metadata_path=val_metadata_path,
                                                  include_metadata=True)

    for batch in train_loader:
        print(batch)
        break
