import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings
import random
import logging
import os
import argparse
from pycocoevalcap.cider.cider import Cider

# warnings coming from loading out of date weights so stopped them for training
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from decoder_lstm import DecoderLSTM
from multimodal_dataloader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    # set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "training_log.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # create dataloaders
    train_loader, val_loader = create_dataloaders(args.batch_size, args.train_features_path, args.val_features_path,
                                                  args.train_metadata_path, args.val_metadata_path,
                                                  include_metadata=True)  # set include_metadata=True

    # subset mode for debug
    if args.subset_mode:
        subset_indices = list(range(100))  # use a subset of 100 samples for testing
        train_loader = DataLoader(Subset(train_loader.dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(val_loader.dataset, subset_indices), batch_size=args.batch_size, shuffle=False)

    # init model
    model = DecoderLSTM(args.feature_dim, args.embedding_dim, args.hidden_dim, args.vocab_size, args.d_model,
                        args.n_head, args.d_k, args.d_v, args.num_layers, args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()  # cross-entropy for caption generation
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # stops model from training when converged for n amount of epochs
    best_val_loss = float('inf')
    patience_counter = 0

    # accuracy metric
    cider_scorer = Cider()

    # train loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in loop:
            image_features = batch['image_feature'].to(device)
            captions = batch['caption_feature'].to(device).long()
            original_captions = batch['caption']  # original captions from batch not features

            # clip captions to ensure all values are within the valid range
            captions = torch.clamp(captions, min=0, max=args.vocab_size - 1)

            # scheduled sampling where model trained on its own predictions gradually
            use_ground_truth = random.random() < args.scheduled_sampling_prob
            if use_ground_truth:
                lstm_input = captions[:, :-1]
            else:
                with torch.no_grad():
                    lstm_input = model(image_features, captions[:, :-1])  # get predictions
                    lstm_input = lstm_input.argmax(dim=-1)  # select the word with the highest probability

            lstm_input = lstm_input.long()

            # forward pass
            outputs = model(image_features, lstm_input)
            loss = criterion(outputs.view(-1, args.vocab_size), captions[:, 1:].reshape(-1))

            # repeating word penalty
            # calculates and normalises the penalty based on the repeated words in the output and batchsize
            penalty = 0
            for i in range(outputs.size(0)):  # loop over the batch
                unique_words, counts = torch.unique(captions[i, 1:], return_counts=True)
                repeated_words = counts[counts > 1]  # finds words that are repeated
                penalty += repeated_words.sum()
            penalty = args.repeat_penalty_weight * (penalty / outputs.size(0))  # normalise by batch size

            # encorporate penalty into loss --> this is why training loss is so high compared to val loss in log
            total_loss = loss + penalty

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        epoch_loss = running_loss / len(train_loader)

        # update scheduled sampling probability (reduce over time to encourage model to generate off of no ground truth)
        args.scheduled_sampling_prob = max(0.1, args.scheduled_sampling_prob * 0.9)

        # val
        model.eval()
        val_loss = 0.0
        val_references = {}
        val_hypotheses = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # load data to device
                image_features = batch['image_feature'].to(device)
                captions = batch['caption_feature'].to(device).long()  # convert to LongTensor for embedding layer
                original_captions = batch['caption']  # original captions from batch

                captions = torch.clamp(captions, min=0, max=args.vocab_size - 1)

                outputs = model(image_features, captions[:, :-1])

                predicted_captions = outputs.argmax(dim=-1)

                # store references and hypotheses for accuracy metric CiDer
                for i in range(len(original_captions)):
                    image_id = f"val_{batch_idx}_{i}"  # create a unique identifier for each sample
                    val_references[image_id] = [original_captions[i]]
                    val_hypotheses[image_id] = [" ".join(map(str, predicted_captions[i].cpu().numpy().tolist()))]  # wrap in a list

                # calc loss
                loss = criterion(outputs.view(-1, args.vocab_size), captions[:, 1:].reshape(-1))
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
            model_save_path = os.path.join(args.log_dir, "best_model.pt")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved at {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered. Training stopped.")
                print("Early stopping triggered. Training stopped.")
                break

    logging.info("Training complete!")
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM for Image Captioning")
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--repeat_penalty_weight', type=float, default=0.1)
    parser.add_argument('--scheduled_sampling_prob', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--train_features_path', type=str, required=True)
    parser.add_argument('--val_features_path', type=str, required=True)
    parser.add_argument('--train_metadata_path', type=str, required=True)
    parser.add_argument('--val_metadata_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--subset_mode', action='store_true')

    args = parser.parse_args()
    train(args)