import torch
import torch.nn as nn

class GeneralModel(nn.Module):
    """Super class to initialize the model"""

    def __init__(self, model_name) :
        super().__init__()
        self.model_name = model_name

    def get_model_name(self):
        """Get the model name """
        return self.model_name