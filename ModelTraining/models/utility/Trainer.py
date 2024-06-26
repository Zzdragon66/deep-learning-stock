import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np


class Trainer():
    """Train the model and save the model with best validation loss"""

    def __init__(self, model, save_dir, loss = torch.nn.MSELoss()):
        # assume the model is the class of GeneralModel
        self.model = model
        self.train_loss_lst, self.val_loss_lst = [], []
        self.min_loss = np.inf
        self.save_dir = Path(save_dir)
        self.loss = loss

        # check the save dir 
        if not self.save_dir.exists():
            raise FileNotFoundError("the save directory is not found")

        # initialize the model directory under the report 
        model_dir : Path = self.save_dir / model.get_model_name()
        if not model_dir.exists():
            model_dir.mkdir(parents =True)

        # intialize the model save filenane here
        self.names  = {
            "weight" : model_dir / "weight.pth", 
            "loss" : model_dir / "loss.json",
            "prediction" : model_dir / "prediction.json",
            "plot" : model_dir/ "comparison_plot.png"
        }

    def train_model(self, train_loader, val_loader, num_epochs = 200, 
                    lr = 1e-3, print_every = 10, if_continue = False):
        """Train the model and save the model with best validation loss in the directory"""
        criterion = self.loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        if not if_continue:
            self.train_loss_lst, self.val_loss_lst = [], []
            self.min_loss = np.inf

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                # adjust for the qunatile loss
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.unsqueeze(-1)
                if output.shape[1] > 1:
                    y_batch = y_batch.expand(-1, output.shape[1])
                # calculate the loss
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step() 
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            self.train_loss_lst.append(avg_train_loss)    
            # check validation accuaracy
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch)
                    # adjust for the qunatile loss
                    if len(y_batch.shape) == 1:
                        y_batch = y_batch.unsqueeze(-1)
                    if output.shape[1] > 1:
                        y_batch = y_batch.expand(-1, output.shape[1])
                    total_val_loss += criterion(output, y_batch).item()
                avg_val_loss = total_val_loss / len(val_loader)
                if avg_val_loss < self.min_loss:
                    self.min_loss = avg_val_loss
                    self.save_model()
                    print("Save model with val loss: ", avg_val_loss)
                self.val_loss_lst.append(avg_val_loss)
                
            if (epoch+1) % print_every == 0:
                print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
                

    def get_best_model(self):
        """Return the model"""
        self.model.load_state_dict(
            torch.load(self.names["weight"])
        )
        return self.model
    
    def get_loss(self):
        """Return the train and validation loss
        Returns: Get the train and validation loss 
        """
        return self.train_loss_lst, self.val_loss_lst
    
    def save_model(self):
        """Save the train loss and validation loss to the save_dir

        Args:
            save_dir (_type_): _description_
        """
        torch.save(
            self.model.state_dict(), 
            self.names["weight"]
        )
    
    def save_loss(self):
        """save the loss list into the save directory

        Args:
            save_dir (Path): the directory to save 
        """
        loss_data = {
            "epochs": list(range(1, len(self.train_loss_lst,) + 1)),
            "train_loss": self.train_loss_lst,
            "valid_loss": self.val_loss_lst
        }
        with open(self.names["loss"], 'w') as file:
            json.dump(loss_data, file)
    
    def save_prediction(self, actual_arr : list, predict_arr : list):
        """Save the prediction at save_dir """
        prediction_data = {
            "actual_list" : list(actual_arr),
            "predict_list" : list(predict_arr)
        }
        with open(self.names["prediction"], "w") as file:
            json.dump(prediction_data, file)

    def save_plot(self, actual_arr : np.array, predict_arr : np.array):
        """Save the plot into the directo·ry"""
        # Assuming predict_arr and actual_arr are your data arrays
        plt.figure(figsize=(14, 7))  # Bigger figure size
        # Plot both sets of data with some transparency
        plt.plot(np.arange(len(predict_arr)), predict_arr, label='Predicted', linestyle='-', color='green', alpha=0.7)
        plt.plot(np.arange(len(actual_arr)), actual_arr, label='Actual', linestyle='--', color='red', alpha=0.7)
        # Add labels, title, and legend
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Comparison of Predictions and Actual Values')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.names["plot"])
        plt.close()  


    def save(self, actual_arr : np.array, predict_arr : np.array, ):
        """Save the prediction and actual array"""
        self.save_loss()
        self.save_prediction(actual_arr, predict_arr)
        self.save_plot(actual_arr, predict_arr)

    
    
 

        