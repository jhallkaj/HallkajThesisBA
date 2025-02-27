import torch
import pandas as pd
import matplotlib.pyplot as plt

from data import Preprocess4Mordred
from model import MordredLinearAutoEncoder
from trainer import MordredTrainer
def main():
    # Data preparation
    preprocessing = Preprocess4Mordred(test_size=0.2, batch_size=8, validation_split=0.1)
    train_load, val_load, test_load = preprocessing.prepare_data(mordred)

    # Model + Trainer initialization
    model = MordredLinearAutoEncoder(mordred.shape[1], hidden = [512,256,32], dropout= 0.3)
    trainer = MordredTrainer(model, learning_rate=0.001, weight_decay= 1e-5)
    
    train_losses = []
    val_losses = []
    test_losses = []


    # Training loop
    epochs = 200 
    for epoch in range(epochs):
        # Training and evaluation
        train_loss = trainer.train(train_load)
        val_loss = trainer.evaluation(val_load)
        test_loss = trainer.evaluation(test_load)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Test Loss: {test_loss:.3f}')

    # Plot the Training Performane and Loss Distribution
    plt.figure(figsize=(10, 6))
    # Train plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.plot(val_losses, label='Validation Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='green')
    plt.title('Training Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.hist(train_losses, alpha=0.5, label='Training', color='red')
    plt.hist(val_losses, alpha=0.5, label='Validation', color='blue')
    plt.hist(test_losses, alpha=0.5, label='Test', color='green')
    plt.title('Loss Distribution')
    plt.xlabel('Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), 'final_mordred_model.pth')


    return model, train_losses, val_losses, test_losses

if __name__ == "__main__":
    model, train_losses, val_losses, test_losses = main()