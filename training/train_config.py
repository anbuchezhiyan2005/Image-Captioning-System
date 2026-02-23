# Step 1: Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim

TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 10,
    'pad_token_id': 0,
    'embedding_dim': 256,
    'hidden_size': 512,
    'num_layers': 1,    
}

def get_config():
    return TRAINING_CONFIG

# Step 2: Define the loss function for training the decoder
def get_loss_function(pad_token_id = 0):
    return nn.CrossEntropyLoss(ignore_index = pad_token_id)

# Step 3: Define the optimizer for training the decoder
def get_optimizer(model_parameters, learning_rate = 0.001):
    return optim.Adam(model_parameters, lr = learning_rate)

# Step 5: Define the main block for testing
if __name__ == "__main__":

    print("="*50)
    print("TESTING TRAINING CONFIGURATION")
    print("="*50)
    
    # Step 6: Test loss function
    print("\n1. Testing Loss Function...")
    loss_fn = get_loss_function(pad_token_id=0)
    print(f"   Loss function: {loss_fn}")
    print(f"   Ignoring token ID: {loss_fn.ignore_index}")
    
    # Create mock data to test loss calculation
    # Predictions: [batch*seq_len, vocab_size]
    mock_predictions = torch.randn(320, 8256)  # 32 batch * 10 seq_len
    # Targets: [batch*seq_len]
    mock_targets = torch.randint(0, 8256, (320,))
    
    loss_value = loss_fn(mock_predictions, mock_targets)
    print(f"   Mock loss value: {loss_value.item():.4f}")
    
    # Step 7: Test optimizer
    print("\n2. Testing Optimizer...")
    # Create a simple model with parameters
    mock_model = nn.Linear(10, 5)
    optimizer = get_optimizer(mock_model.parameters(), learning_rate=0.001)
    print(f"   Optimizer: {optimizer}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    
    print("\n" + "="*50)
    print("✅ TRAINING CONFIGURATION TEST PASSED!")
    print("="*50)