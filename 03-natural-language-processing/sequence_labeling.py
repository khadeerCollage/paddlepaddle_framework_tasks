"""
PaddlePaddle Sequence Labeling Demo
====================================

This script demonstrates Named Entity Recognition (NER) using LSTM.

Task: Identify entities in text
Example: "Apple Inc. is in California" → [B-ORG, I-ORG, O, O, B-LOC]
"""

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader
import numpy as np

# ============================================================================
# DATASET CLASS
# ============================================================================

class SequenceLabelingDataset(Dataset):
    """
    Dataset for sequence labeling tasks (NER, POS tagging, etc.)
    
    Each sample is a sequence of token IDs with corresponding labels.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pad_sequences(sequences, maxlen=None, padding_value=0):
    """
    Pad sequences to the same length
    
    Args:
        sequences: List of lists (variable length sequences)
        maxlen: Maximum length (if None, use longest sequence)
        padding_value: Value to use for padding
    
    Returns:
        Padded numpy array
    """
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded = np.full((len(sequences), maxlen), padding_value, dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded[i, :length] = seq[:length]
    
    return padded


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LSTMSequenceLabelingModel(nn.Layer):
    """
    LSTM-based sequence labeling model
    
    Architecture:
    Input → Embedding → LSTM → FC → Output (per token)
    
    Note: PaddlePaddle LSTM expects input as [batch, seq_len, feature]
    and doesn't use 'batch_first' parameter (it's always batch first)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx=0):
        super(LSTMSequenceLabelingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # PaddlePaddle LSTM is always batch_first by default
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Embedding: [batch, seq_len] → [batch, seq_len, embedding_dim]
        x = self.embedding(x)
        
        # LSTM: [batch, seq_len, embedding_dim] → [batch, seq_len, hidden_dim]
        # PaddlePaddle LSTM returns (output, (h_n, c_n))
        x, (h_n, c_n) = self.lstm(x)
        
        # FC: [batch, seq_len, hidden_dim] → [batch, seq_len, output_dim]
        x = self.fc(x)
        return x


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Train the sequence labeling model"""
    model.train()
    
    print("\n" + "="*70)
    print("TRAINING SEQUENCE LABELING MODEL")
    print("="*70)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for texts, labels in train_loader:
            # Forward pass
            optimizer.clear_grad()
            outputs = model(texts)
            
            # Reshape for loss computation
            # outputs: [batch, seq_len, output_dim] → [batch*seq_len, output_dim]
            # labels: [batch, seq_len] → [batch*seq_len]
            outputs_flat = outputs.reshape([-1, outputs.shape[-1]])
            labels_flat = labels.reshape([-1])
            
            loss = criterion(outputs_flat, labels_flat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')
    
    print("="*70)
    print("✓ Training Complete!\n")


# ============================================================================
# INFERENCE & VISUALIZATION
# ============================================================================

def predict_and_visualize(model, texts, labels, id2word, id2label):
    """
    Run inference and visualize predictions
    
    Args:
        model: Trained model
        texts: Input token IDs
        labels: True labels
        id2word: Mapping from ID to word
        id2label: Mapping from ID to label
    """
    model.eval()
    
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    with paddle.no_grad():
        outputs = model(texts)
        predictions = paddle.argmax(outputs, axis=-1)
    
    # Calculate accuracy
    correct = 0
    total = 0
    
    # Visualize each sequence
    for i in range(len(texts)):
        print(f"\n📝 Sequence {i+1}:")
        print("-" * 70)
        
        # Header
        print(f"{'Token':<15} {'True Label':<15} {'Predicted':<15} {'Confidence':<10} {'Match'}")
        print("-" * 70)
        
        seq_len = texts.shape[1]
        for j in range(seq_len):
            token_id = int(texts[i][j].numpy())
            
            # Skip padding tokens
            if token_id == 0:
                continue
                
            true_label = int(labels[i][j].numpy())
            pred_label = int(predictions[i][j].numpy())
            
            # Get confidence (softmax probability)
            probs = paddle.nn.functional.softmax(outputs[i][j], axis=-1)
            confidence = float(probs[pred_label].numpy())
            
            # Map IDs to readable text
            word = id2word.get(token_id, f"UNK_{token_id}")
            true_tag = id2label.get(true_label, f"LABEL_{true_label}")
            pred_tag = id2label.get(pred_label, f"LABEL_{pred_label}")
            
            # Check if correct
            match = "✓" if true_label == pred_label else "✗"
            
            # Count accuracy
            if token_id != 0:  # Don't count padding
                total += 1
                if true_label == pred_label:
                    correct += 1
            
            print(f"{word:<15} {true_tag:<15} {pred_tag:<15} {confidence:.4f}     {match}")
    
    # Print overall accuracy
    if total > 0:
        accuracy = (correct / total) * 100
        print("\n" + "-" * 70)
        print(f"Overall Token Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """
    Complete sequence labeling demo with realistic NER example
    """
    print("\n" + "="*70)
    print("🧠 PADDLEPADDLE SEQUENCE LABELING DEMO")
    print("="*70)
    print("\nTask: Named Entity Recognition (NER)")
    print("Goal: Identify organization (ORG) and location (LOC) entities\n")
    
    # ========================================
    # 1. CREATE REALISTIC NER DATASET
    # ========================================
    
    # Vocabulary (word to ID mapping)
    word2id = {
        "<PAD>": 0, "Apple": 1, "Inc.": 2, "is": 3, 
        "located": 4, "in": 5, "California": 6,
        "Microsoft": 7, "Corp": 8, "from": 9, "Seattle": 10,
        "Google": 11, "based": 12, "Mountain": 13, "View": 14
    }
    id2word = {v: k for k, v in word2id.items()}
    
    # Labels (BIO tagging scheme)
    label2id = {
        "O": 0,      # Outside any entity (also used for padding)
        "B-ORG": 1,  # Beginning of Organization
        "I-ORG": 2,  # Inside Organization
        "B-LOC": 3,  # Beginning of Location
        "I-LOC": 4   # Inside Location
    }
    id2label = {v: k for k, v in label2id.items()}
    
    # Training examples
    # Format: [word_ids], [label_ids]
    examples = [
        # "Apple Inc. is located in California"
        ([1, 2, 3, 4, 5, 6], [1, 2, 0, 0, 0, 3]),
        
        # "Microsoft Corp from Seattle"
        ([7, 8, 9, 10], [1, 2, 0, 3]),
        
        # "Google based in Mountain View"
        ([11, 12, 5, 13, 14], [1, 0, 0, 3, 4]),
        
        # More training examples (repeated for better learning)
        ([1, 2, 5, 6], [1, 2, 0, 3]),
        ([7, 8, 5, 10], [1, 2, 0, 3]),
        ([11, 5, 13, 14], [1, 0, 3, 4]),
        
        # Additional examples
        ([1, 2], [1, 2]),  # Just "Apple Inc."
        ([7, 8], [1, 2]),  # Just "Microsoft Corp"
        ([11], [1]),        # Just "Google"
    ]
    
    print("📊 Dataset Statistics:")
    print(f"   - Number of examples: {len(examples)}")
    print(f"   - Vocabulary size: {len(word2id)}")
    print(f"   - Number of labels: {len(label2id)}")
    print(f"   - Label scheme: BIO (Begin, Inside, Outside)")
    
    # Show example
    print("\n📝 Example Training Sentence:")
    example_idx = 0
    example_words = [id2word[i] for i in examples[example_idx][0]]
    example_labels = [id2label[i] for i in examples[example_idx][1]]
    print(f"   Words:  {' '.join(example_words)}")
    print(f"   Labels: {' '.join(example_labels)}")
    
    # ========================================
    # 2. PAD SEQUENCES AND CREATE TENSORS
    # ========================================
    
    print("\n🔧 Padding sequences to uniform length...")
    
    # Extract texts and labels
    text_sequences = [ex[0] for ex in examples]
    label_sequences = [ex[1] for ex in examples]
    
    # Find max length
    max_len = max(len(seq) for seq in text_sequences)
    print(f"   - Maximum sequence length: {max_len}")
    
    # Pad sequences
    texts_padded = pad_sequences(text_sequences, maxlen=max_len, padding_value=0)
    labels_padded = pad_sequences(label_sequences, maxlen=max_len, padding_value=0)
    
    # Convert to tensors
    texts = paddle.to_tensor(texts_padded, dtype='int64')
    labels = paddle.to_tensor(labels_padded, dtype='int64')
    
    print(f"   - Padded tensor shape: {texts.shape}")
    
    # ========================================
    # 3. CREATE DATALOADER
    # ========================================
    
    dataset = SequenceLabelingDataset(texts, labels)
    train_loader = DataLoader(dataset, batch_size=3, shuffle=True)
    
    # ========================================
    # 4. INITIALIZE MODEL
    # ========================================
    
    # Hyperparameters
    vocab_size = len(word2id)
    embedding_dim = 32
    hidden_dim = 64
    output_dim = len(label2id)
    num_epochs = 30  # More epochs for better learning
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   - Vocabulary size: {vocab_size}")
    print(f"   - Embedding dimension: {embedding_dim}")
    print(f"   - LSTM hidden dimension: {hidden_dim}")
    print(f"   - Output classes: {output_dim}")
    
    model = LSTMSequenceLabelingModel(
        vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx=0
    )
    
    # Count parameters
    total_params = sum(p.numel().item() for p in model.parameters())
    print(f"   - Total parameters: {total_params:,}")
    
    # Use ignore_index=0 to ignore padding in loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(learning_rate=0.01, parameters=model.parameters())
    
    # ========================================
    # 5. TRAIN MODEL
    # ========================================
    
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # ========================================
    # 6. TEST & VISUALIZE
    # ========================================
    
    # Test on training data (for demonstration)
    print("\n" + "="*70)
    print("📊 EVALUATION ON TRAINING DATA")
    print("="*70)
    predict_and_visualize(model, texts, labels, id2word, id2label)
    
    # ========================================
    # 7. TEST ON NEW SENTENCE
    # ========================================
    
    print("\n" + "="*70)
    print("🧪 TESTING ON NEW SENTENCE")
    print("="*70)
    
    # New test sentence: "Apple is in California"
    test_text_seq = [[1, 3, 5, 6]]  # Apple is in California
    test_label_seq = [[1, 0, 0, 3]]  # B-ORG O O B-LOC
    
    # Pad to same length
    test_text_padded = pad_sequences(test_text_seq, maxlen=max_len, padding_value=0)
    test_label_padded = pad_sequences(test_label_seq, maxlen=max_len, padding_value=0)
    
    test_text = paddle.to_tensor(test_text_padded, dtype='int64')
    test_label = paddle.to_tensor(test_label_padded, dtype='int64')
    
    print("\nTest Sentence: 'Apple is in California'")
    print("Expected: Apple(B-ORG) is(O) in(O) California(B-LOC)")
    predict_and_visualize(model, test_text, test_label, id2word, id2label)
    
    # ========================================
    # 8. SUMMARY
    # ========================================
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n📚 Summary:")
    print("   - Implemented LSTM-based NER model")
    print("   - Trained on organization and location entities")
    print("   - Achieved token-level classification")
    print("   - Used BIO tagging scheme")
    print("   - Handled variable-length sequences with padding")
    print("\n💡 Key Concepts:")
    print("   - Sequence labeling assigns labels to each token")
    print("   - Different from text classification (one label per document)")
    print("   - LSTM captures contextual information")
    print("   - Loss computed per token, then averaged")
    print("   - Padding tokens ignored in loss computation")
    print("\n🎯 BIO Tagging Scheme:")
    print("   - B-XXX: Beginning of entity type XXX")
    print("   - I-XXX: Inside (continuation) of entity type XXX")
    print("   - O: Outside any entity")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()