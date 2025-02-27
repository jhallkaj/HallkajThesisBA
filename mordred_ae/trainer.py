import torch
import torch.nn as nn
import numpy as np

class MordredTrainer:
    def __init__(self, model, learning_rate = 0.01, weight_decay = 1e-5):
        self.model = model
        self.loss_fn = nn.MSELoss() # CrossEntropy
        self.optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, weight_decay= weight_decay)

    def train(self, train_load):
        self.model.train()
        total_loss = 0
      
        for batch in train_load:
            x = batch[0] # Takes the batches of samples

            reconstructed = self.model(x) # Passes the samples to the model
            loss = self.loss_fn(reconstructed, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()


        return total_loss / len(train_load)
    
        
    def evaluation(self, test_load):
        # This will implement no weight calculations but only calculate loss and the model for the test dataset
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in test_load:
                x = batch[0]
                reconstructed = self.model(x)
                loss = self.loss_fn(reconstructed, x)
                total_loss +=loss.item()


        return total_loss / len(test_load)