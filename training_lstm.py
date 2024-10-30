import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings
import random
import logging
import os
from pycocoevalcap.cider.cider import Cider

# warnings coming from loading out of date weights so stopped them for training
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from decoder_lstm import DecoderLSTM
from multimodal_dataloader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(feature_dim, embedding_dim, hidden_dim, vocab_size, d_model, n_head, d_k, d_v, num_layers, dropout,
          learning_rate, num_epochs, batch_size, repeat_penalty_weight, scheduled_sampling_prob, patience,
          train_features_path, val_features_path, train_metadata_path, val_metadata_path, log_dir, subset_mode=False):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # create dataloaders
    train_loader, val_loader = create_dataloaders(batch_size, train_features_path, val_features_path,
                                                  train_metadata_path, val_metadata_path,
                                                  include_metadata=True)  # Set include_metadata=True

    # subset mode for debug
    if subset_mode:
        subset_indices = list(range(100))
        train_loader = DataLoader(Subset(train_loader.dataset, subset_indices), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(val_loader.dataset, subset_indices), batch_size=batch_size, shuffle=False)

    # init model
    model = DecoderLSTM(feature_dim, embedding_dim, hidden_dim, vocab_size, d_model, n_head, d_k, d_v, num_layers,
                        dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # stops model from training when converged for n amount of epochs
    best_val_loss = float('inf')
    patience_counter = 0

    # accuracy metric
    cider_scorer = Cider()

    # train loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in loop:
            image_features = batch['image_feature'].to(device)
            captions = batch['caption_feature'].to(device).long()
            original_captions = batch['caption']  # original captions from batch not features

            # clip captions to ensure all values are within the valid range
            captions = torch.clamp(captions, min=0, max=vocab_size - 1)

            # scheduled sampling where model trained on its own predictions gradually
            use_ground_truth = random.random() < scheduled_sampling_prob
            if use_ground_truth:
                lstm_input = captions[:, :-1]
            else:
                with torch.no_grad():
                    lstm_input = model(image_features, captions[:, :-1])
                    lstm_input = lstm_input.argmax(dim=-1)

            lstm_input = lstm_input.long()

            # forward pass
            outputs = model(image_features, lstm_input)
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))

            # repeating word penalty
            # calculates and normalises the penalty based on the repeated words in the output and batchsize
            penalty = 0
            for i in range(outputs.size(0)):  # Loop over the batch
                unique_words, counts = torch.unique(captions[i, 1:], return_counts=True)
                repeated_words = counts[counts > 1]  # finds words that are repeated
                penalty += repeated_words.sum()
            penalty = repeat_penalty_weight * (penalty / outputs.size(0))  # normalise by batch size

            # encorporate penalty into loss --> this is why training loss is so high compared to val loss in log
            total_loss = loss + penalty


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        epoch_loss = running_loss / len(train_loader)

        # update scheduled sampling probability (reduce over time to encourage model to generate off of no ground truth)
        scheduled_sampling_prob = max(0.1, scheduled_sampling_prob * 0.9)

        # val
        model.eval()
        val_loss = 0.0
        val_references = {}
        val_hypotheses = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Load data to device
                image_features = batch['image_feature'].to(device)
                captions = batch['caption_feature'].to(device).long()  # convert to LongTensor for embedding layer
                original_captions = batch['caption']  # original captions from batch

                captions = torch.clamp(captions, min=0, max=vocab_size - 1)

                outputs = model(image_features, captions[:, :-1])

                predicted_captions = outputs.argmax(dim=-1)

                # store references and hypotheses for accuracy metric CiDer
                for i in range(len(original_captions)):
                    image_id = f"val_{batch_idx}_{i}"  # create a unique identifier for each sample
                    val_references[image_id] = [original_captions[i]]
                    val_hypotheses[image_id] = [" ".join(map(str, predicted_captions[i].cpu().numpy().tolist()))]  # wrap in a list

                # calc loss
                loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # calc CIDEr score
        cider_score, _ = cider_scorer.compute_score(val_references, val_hypotheses)

        log_message = f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, CIDEr Score: {cider_score:.4f}"
        logging.info(log_message)
        print(log_message)

        # save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_save_path = os.path.join(log_dir, "best_model.pt")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved at {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered. Training stopped.")
                print("Early stopping triggered. Training stopped.")
                break

    logging.info("Training complete!")
    print("Training complete!")

if __name__ == "__main__":
    train(
        feature_dim=512,
        embedding_dim=256,
        hidden_dim=512,
        vocab_size=10000,
        d_model=512,
        n_head=8,
        d_k=64,
        d_v=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=1e-4,
        num_epochs=5,
        batch_size=16,
        repeat_penalty_weight=0.1,
        scheduled_sampling_prob=0.5,
        patience=3,
        train_features_path=r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\flickr30k features\train_extracted_features.pt",
        val_features_path=r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\flickr30k features\val_extracted_features.pt",
        train_metadata_path=r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\train_dataset.pt",
        val_metadata_path=r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\datasets\augmented flickr30k\val_dataset.pt",
        log_dir=r"C:\Users\charl\PycharmProjects\Image-Auto-Captioner\Image-Auto-Captioning\model_details",
        subset_mode=True  # Enable subset mode for testing
    )
