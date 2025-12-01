"""
PaddlePaddle Text Classification Demo
======================================

This script demonstrates sentiment analysis using LSTM for text classification.

Task: Classify movie reviews as Positive or Negative
"""

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader
import numpy as np

# ============================================================================
# DATASET CLASS
# ============================================================================

class TextDataset(Dataset):
    """
    Dataset for text classification tasks
    
    Each sample is a sequence of token IDs with a single label.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LSTMTextClassifier(nn.Layer):
    """
    LSTM-based text classification model
    
    Architecture:
    Input → Embedding → LSTM → Take last hidden → FC → Output
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, padding_idx=0):
        super(LSTMTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Embedding: [batch, seq_len] → [batch, seq_len, embed_size]
        x = self.embedding(x)
        
        # LSTM: [batch, seq_len, embed_size] → [batch, seq_len, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state: [batch, hidden_size]
        last_hidden = lstm_out[:, -1, :]
        
        # Dropout for regularization
        x = self.dropout(last_hidden)
        
        # FC: [batch, hidden_size] → [batch, num_classes]
        x = self.fc(x)
        return x


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pad_sequences(sequences, maxlen=None, padding_value=0):
    """Pad sequences to uniform length"""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded = np.full((len(sequences), maxlen), padding_value, dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded[i, :length] = seq[:length]
    
    return padded


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Train the text classification model"""
    model.train()
    
    print("\n" + "="*70)
    print("TRAINING TEXT CLASSIFICATION MODEL")
    print("="*70)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        
        for texts, labels in train_loader:
            optimizer.clear_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = paddle.argmax(outputs, axis=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        accuracy = (correct / total) * 100
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print("="*70)
    print("✓ Training Complete!\n")


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, test_loader, id2word, id2label):
    """Evaluate and visualize predictions"""
    model.eval()
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    all_predictions = []
    all_labels = []
    all_texts = []
    
    with paddle.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            predictions = paddle.argmax(outputs, axis=1)
            probs = paddle.nn.functional.softmax(outputs, axis=1)
            
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
            all_texts.extend(texts.numpy())
    
    # Calculate accuracy
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = (correct / len(all_labels)) * 100
    
    print(f"\n📊 Overall Accuracy: {accuracy:.2f}% ({correct}/{len(all_labels)})")
    print("\n" + "-"*70)
    print("Sample Predictions:")
    print("-"*70)
    print(f"{'Text':<40} {'True':<10} {'Predicted':<10} {'Match'}")
    print("-"*70)
    
    # Show first 10 examples
    for i in range(min(10, len(all_texts))):
        text_ids = all_texts[i]
        # Convert to words (show first 5 tokens)
        words = [id2word.get(int(tid), "<PAD>") for tid in text_ids[:5] if tid != 0]
        text_str = " ".join(words)
        if len(text_str) > 35:
            text_str = text_str[:32] + "..."
        
        true_label = id2label[all_labels[i]]
        pred_label = id2label[all_predictions[i]]
        match = "✓" if all_labels[i] == all_predictions[i] else "✗"
        
        print(f"{text_str:<40} {true_label:<10} {pred_label:<10} {match}")
    
    print("="*70)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """
    Complete text classification demo with sentiment analysis
    """
    print("\n" + "="*70)
    print("🎬 PADDLEPADDLE TEXT CLASSIFICATION DEMO")
    print("="*70)
    print("\nTask: Sentiment Analysis")
    print("Goal: Classify movie reviews as Positive (1) or Negative (0)\n")
    
    # ========================================
    # 1. CREATE SENTIMENT DATASET
    # ========================================
    
    # Vocabulary
    word2id = {
        "<PAD>": 0, "great": 1, "excellent": 2, "amazing": 3, "love": 4,
        "best": 5, "wonderful": 6, "fantastic": 7, "good": 8,
        "terrible": 9, "awful": 10, "bad": 11, "worst": 12, "hate": 13,
        "boring": 14, "disappointing": 15, "movie": 16, "film": 17,
        "this": 18, "is": 19, "the": 20, "a": 21, "very": 22
    }
    id2word = {v: k for k, v in word2id.items()}
    
    # Labels
    label2id = {"Negative": 0, "Positive": 1}
    id2label = {0: "Negative", 1: "Positive"}
    
    # Training examples (token_ids, label)
    # Positive reviews
    positive_reviews = [
        [18, 16, 19, 1],           # "this movie is great"
        [2, 17, 22, 8],            # "excellent film very good"
        [4, 18, 3, 16],            # "love this amazing movie"
        [6, 7, 5, 17],             # "wonderful fantastic best film"
        [18, 19, 20, 5, 16],       # "this is the best movie"
        [1, 17, 4, 19],            # "great film love is"
        [3, 2, 6, 16],             # "amazing excellent wonderful movie"
        [22, 1, 7, 17],            # "very great fantastic film"
    ]
    
    # Negative reviews
    negative_reviews = [
        [18, 16, 19, 9],           # "this movie is terrible"
        [10, 17, 22, 11],          # "awful film very bad"
        [13, 18, 12, 16],          # "hate this worst movie"
        [14, 15, 11, 17],          # "boring disappointing bad film"
        [18, 19, 20, 12, 16],      # "this is the worst movie"
        [9, 17, 13, 19],           # "terrible film hate is"
        [11, 10, 14, 16],          # "bad awful boring movie"
        [22, 9, 15, 17],           # "very terrible disappointing film"
    ]
    
    # Create dataset
    text_sequences = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    # Duplicate data for better training
    text_sequences = text_sequences * 10
    labels = labels * 10
    
    print("📊 Dataset Statistics:")
    print(f"   - Number of examples: {len(text_sequences)}")
    print(f"   - Positive examples: {sum(labels)}")
    print(f"   - Negative examples: {len(labels) - sum(labels)}")
    print(f"   - Vocabulary size: {len(word2id)}")
    
    # Show examples
    print("\n📝 Sample Reviews:")
    print(f"   Positive: \"{' '.join([id2word[i] for i in positive_reviews[0]])}\"")
    print(f"   Negative: \"{' '.join([id2word[i] for i in negative_reviews[0]])}\"")
    
    # ========================================
    # 2. PAD SEQUENCES
    # ========================================
    
    print("\n🔧 Preparing data...")
    max_len = max(len(seq) for seq in text_sequences)
    print(f"   - Maximum sequence length: {max_len}")
    
    texts_padded = pad_sequences(text_sequences, maxlen=max_len, padding_value=0)
    labels_array = np.array(labels, dtype=np.int64)
    
    texts = paddle.to_tensor(texts_padded, dtype='int64')
    labels_tensor = paddle.to_tensor(labels_array, dtype='int64')
    
    print(f"   - Padded tensor shape: {texts.shape}")
    
    # ========================================
    # 3. CREATE DATALOADER
    # ========================================
    
    dataset = TextDataset(texts, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # ========================================
    # 4. INITIALIZE MODEL
    # ========================================
    
    vocab_size = len(word2id)
    embed_size = 32
    hidden_size = 64
    num_classes = 2
    num_epochs = 20
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   - Vocabulary size: {vocab_size}")
    print(f"   - Embedding dimension: {embed_size}")
    print(f"   - LSTM hidden size: {hidden_size}")
    print(f"   - Output classes: {num_classes}")
    
    model = LSTMTextClassifier(vocab_size, embed_size, hidden_size, num_classes, padding_idx=0)
    
    total_params = sum(p.numel().item() for p in model.parameters())
    print(f"   - Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(learning_rate=0.01, parameters=model.parameters())
    
    # ========================================
    # 5. TRAIN MODEL
    # ========================================
    
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # ========================================
    # 6. EVALUATE
    # ========================================
    
    evaluate_model(model, train_loader, id2word, id2label)
    
    # ========================================
    # 7. TEST NEW REVIEWS
    # ========================================
    
    print("\n" + "="*70)
    print("🧪 TESTING ON NEW REVIEWS")
    print("="*70)
    
    test_reviews = [
        ([1, 3, 16], "great amazing movie", 1),  # Positive
        ([9, 11, 17], "terrible bad film", 0),    # Negative
        ([5, 6, 17], "best wonderful film", 1),   # Positive
        ([12, 10, 16], "worst awful movie", 0),   # Negative
    ]
    
    model.eval()
    print("\n")
    for tokens, text, true_label in test_reviews:
        # Pad
        padded = pad_sequences([tokens], maxlen=max_len, padding_value=0)
        test_tensor = paddle.to_tensor(padded, dtype='int64')
        
        with paddle.no_grad():
            output = model(test_tensor)
            probs = paddle.nn.functional.softmax(output, axis=1)
            prediction = paddle.argmax(output, axis=1).item()
            confidence = probs[0][prediction].item()
        
        pred_label = id2label[prediction]
        true_label_str = id2label[true_label]
        match = "✓" if prediction == true_label else "✗"
        
        print(f"Review: \"{text}\"")
        print(f"   True: {true_label_str}, Predicted: {pred_label}, Confidence: {confidence:.2%} {match}\n")
    
    # ========================================
    # 8. SUMMARY
    # ========================================
    
    print("="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n📚 Summary:")
    print("   - Implemented LSTM-based sentiment classifier")
    print("   - Trained on positive and negative movie reviews")
    print("   - Achieved document-level classification")
    print("   - Used word embeddings and LSTM for context")
    print("\n💡 Key Differences from Sequence Labeling:")
    print("   - Text Classification: ONE label per document")
    print("   - Sequence Labeling: ONE label per token")
    print("   - Architecture: Uses last LSTM hidden state")
    print("   - Application: Sentiment analysis, topic classification")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()