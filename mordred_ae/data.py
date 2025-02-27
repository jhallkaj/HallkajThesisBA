import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class Preprocess4Mordred:
    def __init__(self, test_size = 0.2, batch_size = 16, validation_split = 0.1):
        # We initialize the important things
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.batch_size = batch_size
        self.validation_split = validation_split
    
    def prepare_data(self, mordred):
        # Convert mordred to np.array, scale it, and split into train and test
        mordred_array = mordred.values # convert to np.array
        mordred_array = self.scaler.fit_transform(mordred_array) # scale our data

        train_val_data, test_data = train_test_split(mordred_array, test_size=self.test_size, random_state= 42) # first we split the test data
        train_data, val_data = train_test_split(train_val_data, test_size=self.validation_split, random_state= 42) # then we split the validation data


        # Convert to Tensors and create the datasets with tensor wrapping, one tensor only
        X_train = torch.FloatTensor(train_data)  # convert to float type for PyTOrch
        X_val = torch.FloatTensor(val_data)
        X_test = torch.FloatTensor(test_data)
        train_dataset = torch.utils.data.TensorDataset(X_train) # combines them into a dataset
        val_dataset = torch.utils.data.TensorDataset(X_val)
        test_dataset = torch.utils.data.TensorDataset(X_test)

        # Standard Data Loader
        train_load = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True)
        val_load = torch.utils.data.DataLoader(val_dataset)
        test_load = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False) # load the data

        return train_load, val_load, test_load # return what we will use for our training