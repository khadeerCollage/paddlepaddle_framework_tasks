"""
PaddlePaddle Transformer Tutorial
==================================

This script demonstrates the Transformer architecture for sequence classification.

Task: Language Understanding with Self-Attention
"""

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
import numpy as np
import math

# ============================================================================
# DATASET CLASS
# ============================================================================

class TransformerDataset(Dataset):
    """Dataset for transformer-based sequence classification"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Layer):
    """
    Adds positional information to embeddings
    
    Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2, dtype='float32') * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        return x + self.pe[:, :x.shape[1], :]


# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Layer):
    """
    Multi-Head Self-Attention mechanism
    
    Allows model to focus on different parts of the sequence simultaneously
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """Split into multiple heads"""
        batch_size = x.shape[0]
        x = x.reshape([batch_size, -1, self.num_heads, self.d_k])
        return x.transpose([0, 2, 1, 3])  # [batch, heads, seq, d_k]
    
    def forward(self, x):
        """
        Compute multi-head attention
        
        Input: [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
        """
        batch_size = x.shape[0]
        
        # Linear projections
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        # Scaled dot-product attention
        scores = paddle.matmul(Q, K.transpose([0, 1, 3, 2])) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, axis=-1)
        attn_output = paddle.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, -1, self.d_model])
        
        # Final linear projection
        output = self.W_o(attn_output)
        return output


# ============================================================================
# FEED-FORWARD NETWORK
# ============================================================================

class FeedForward(nn.Layer):
    """
    Position-wise feed-forward network
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================================
# TRANSFORMER ENCODER LAYER
# ============================================================================

class TransformerEncoderLayer(nn.Layer):
    """
    Single Transformer encoder layer
    
    Components:
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass through encoder layer"""
        # Multi-head attention with residual
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# ============================================================================
# COMPLETE TRANSFORMER MODEL
# ============================================================================

class TransformerClassifier(nn.Layer):
    """
    Transformer-based sequence classifier
    
    Architecture:
    Input → Embedding → Positional Encoding → Encoder Layers → Pool → Classify
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, num_classes, max_len=512):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Stack of encoder layers
        self.encoder_layers = nn.LayerList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Forward pass
        
        Input: [batch, seq_len]
        Output: [batch, num_classes]
        """
        # Embedding + Positional Encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global average pooling
        x = paddle.mean(x, axis=1)
        
        # Classification
        x = self.fc(x)
        return x


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Train the transformer model"""
    model.train()
    
    print("\n" + "="*70)
    print("TRAINING TRANSFORMER MODEL")
    print("="*70)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            optimizer.clear_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = paddle.argmax(outputs, axis=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.shape[0]
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        accuracy = (correct / total) * 100
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print("="*70)
    print("✓ Training Complete!\n")


def evaluate_model(model, test_loader, id2word, id2label):
    """Evaluate transformer model"""
    model.eval()
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    correct = 0
    total = 0
    
    with paddle.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            predictions = paddle.argmax(outputs, axis=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
    
    accuracy = (correct / total) * 100
    print(f"\n📊 Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*70)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Complete transformer demo"""
    print("\n" + "="*70)
    print("🤖 PADDLEPADDLE TRANSFORMER TUTORIAL")
    print("="*70)
    print("\nTask: Sequence Classification with Self-Attention")
    print("Goal: Demonstrate modern Transformer architecture\n")
    
    # ========================================
    # 1. CREATE DATASET
    # ========================================
    
    # Vocabulary
    word2id = {
        "<PAD>": 0, "good": 1, "great": 2, "excellent": 3, "amazing": 4,
        "bad": 5, "terrible": 6, "awful": 7, "poor": 8,
        "movie": 9, "film": 10, "show": 11, "video": 12,
        "love": 13, "hate": 14, "like": 15, "dislike": 16,
        "best": 17, "worst": 18, "nice": 19, "boring": 20
    }
    id2word = {v: k for k, v in word2id.items()}
    
    label2id = {"Negative": 0, "Positive": 1}
    id2label = {0: "Negative", 1: "Positive"}
    
    # Create synthetic dataset
    positive_samples = [
        [1, 2, 9], [3, 4, 10], [13, 17, 11], [2, 19, 12],
        [1, 15, 10], [4, 3, 9], [17, 2, 11], [13, 1, 10]
    ]
    negative_samples = [
        [5, 6, 9], [7, 8, 10], [14, 18, 11], [6, 20, 12],
        [5, 16, 10], [8, 7, 9], [18, 6, 11], [14, 5, 10]
    ]
    
    # Pad sequences
    max_len = 10
    def pad_seq(seq, maxlen):
        padded = [0] * maxlen
        for i, val in enumerate(seq[:maxlen]):
            padded[i] = val
        return padded
    
    all_sequences = [pad_seq(s, max_len) for s in positive_samples + negative_samples]
    all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    # Duplicate for training
    all_sequences = all_sequences * 50
    all_labels = all_labels * 50
    
    print("📊 Dataset Statistics:")
    print(f"   - Total samples: {len(all_sequences)}")
    print(f"   - Sequence length: {max_len}")
    print(f"   - Vocabulary size: {len(word2id)}")
    print(f"   - Classes: {len(label2id)}")
    
    # Convert to tensors
    data = paddle.to_tensor(all_sequences, dtype='int64')
    labels = paddle.to_tensor(all_labels, dtype='int64')
    
    # ========================================
    # 2. CREATE DATALOADER
    # ========================================
    
    dataset = TransformerDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # ========================================
    # 3. INITIALIZE MODEL
    # ========================================
    
    vocab_size = len(word2id)
    d_model = 64      # Model dimension
    num_heads = 4     # Number of attention heads
    num_layers = 2    # Number of encoder layers
    d_ff = 256        # Feed-forward dimension
    num_classes = 2
    num_epochs = 20
    
    print(f"\n🏗️  Transformer Architecture:")
    print(f"   - Model dimension (d_model): {d_model}")
    print(f"   - Number of attention heads: {num_heads}")
    print(f"   - Number of encoder layers: {num_layers}")
    print(f"   - Feed-forward dimension: {d_ff}")
    print(f"   - Output classes: {num_classes}")
    
    model = TransformerClassifier(
        vocab_size, d_model, num_heads, num_layers, d_ff, num_classes, max_len
    )
    
    total_params = sum(p.numel().item() for p in model.parameters())
    print(f"   - Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(learning_rate=0.001, parameters=model.parameters())
    
    # ========================================
    # 4. TRAIN MODEL
    # ========================================
    
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # ========================================
    # 5. EVALUATE
    # ========================================
    
    evaluate_model(model, train_loader, id2word, id2label)
    
    # ========================================
    # 6. TEST ATTENTION MECHANISM
    # ========================================
    
    print("\n" + "="*70)
    print("🔍 TESTING SELF-ATTENTION")
    print("="*70)
    
    test_samples = [
        ([2, 3, 9], "great excellent movie", 1),
        ([6, 7, 10], "terrible awful film", 0),
        ([17, 4, 11], "best amazing show", 1),
    ]
    
    model.eval()
    print("\n")
    for tokens, text, true_label in test_samples:
        padded = pad_seq(tokens, max_len)
        test_tensor = paddle.to_tensor([padded], dtype='int64')
        
        with paddle.no_grad():
            output = model(test_tensor)
            probs = F.softmax(output, axis=1)
            prediction = paddle.argmax(output, axis=1).item()
            confidence = probs[0][prediction].item()
        
        pred_label = id2label[prediction]
        true_label_str = id2label[true_label]
        match = "✓" if prediction == true_label else "✗"
        
        print(f"Text: \"{text}\"")
        print(f"   True: {true_label_str}, Predicted: {pred_label}")
        print(f"   Confidence: {confidence:.2%} {match}\n")
    
    # ========================================
    # 7. SUMMARY
    # ========================================
    
    print("="*70)
    print("✅ TRANSFORMER TUTORIAL COMPLETE!")
    print("="*70)
    print("\n📚 Key Concepts Covered:")
    print("   ✓ Self-Attention Mechanism")
    print("   ✓ Multi-Head Attention")
    print("   ✓ Positional Encoding")
    print("   ✓ Encoder Layer (Attention + FFN)")
    print("   ✓ Residual Connections & Layer Normalization")
    print("\n💡 Advantages of Transformers:")
    print("   • Parallel processing (faster than RNNs)")
    print("   • Long-range dependencies captured directly")
    print("   • State-of-the-art performance on NLP tasks")
    print("   • Foundation of modern models (BERT, GPT)")
    print("\n🎯 Architecture Comparison:")
    print("   LSTM:        Sequential processing, slower")
    print("   Transformer: Parallel attention, faster & better")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()