import torch
import torch.nn as nn
import torchinfo
import torchvision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

##############################################################
# Utils
##############################################################

def print_model_summary(model, 
                        device:str = 'cpu',
                        input_dim:tuple = (32, 3, 224, 224)):
    """
    Function which prints summary of the model.

    Args:
        * model, pytorch model
        * device, str, where the model and data are located, default is: cpu
        * input_dim, tupple, input dimenzions to the model

    Output:
        * returns printable model summary
    """
    return torchinfo.summary(model=model, 
            input_size= input_dim, # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=16,
            device = device,
            row_settings=["var_names"]
            ) 


##############################################################
# Models --> modules for shrniking
##############################################################
class Shrink_Module_1x1Conv2D(nn.Module):
    def __init__(self, in_channels: int = 224):
        """
        Method for shrinking with 1x1 conv layers
        """
        super().__init__()

        self.coronal_shrink = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.axial_shrink = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sagittal_shrink = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

        # Define cat layer
        self.concatenation_layer = CatLayer(dim = 1) 

    def forward(self, x):
        # Shrink to one channel by calculatin mean
        _x_coronal = self.coronal_shrink(x)
        _x_axial = self.axial_shrink(x)
        _x_sagittal = self.sagittal_shrink(x)
        
        # Concat
        _x = self.concatenation_layer((_x_coronal, _x_axial, _x_sagittal))
        return _x

class Full_Shrink_1x1Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        _x = torch.squeeze(x, 1)
        
        # First dimension
        _x = self.conv1x1(_x)
        
        # Second dimension
        _x = _x.transpose(1, 3).flip(3)
        _x = self.conv1x1(_x)
        
        # Third dimension
        _x = _x.transpose(1, 2).flip(2)
        _x = self.conv1x1(_x)  

        # Revert
        _x = _x.flip(2).transpose(1, 2)
        _x = _x.flip(3).transpose(1, 3)
        return _x


# In order to display mean layers
class MeanLayer(nn.Module):
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)

class CatLayer(nn.Module):
    def __init__(self, dim: int = 1):
        super(CatLayer, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim = 1)
    
class PaddingLayer(nn.Module):
    def __init__(self, output_dim: int = 224):
        """
        Function which applies padding
        Args:
            * output_dim, integer, desired output dim for each dimension
        """
        super(PaddingLayer, self).__init__()
        self.output_dim = output_dim

    def forward(self, x):
        """
        Forward pass of a module
        """
        _, _, _x_dim, _y_dim = x.shape
        if self.output_dim < _x_dim or self.output_dim < _y_dim:
            x = self.__interpolate_with_aspect_ratio(x, self.output_dim)
            _, _, _x_dim, _y_dim = x.shape
        _x = F.pad(x, 
                   self.__obtain_padding((_x_dim, _y_dim)), 
                   mode='constant', 
                   value=0)
        return _x

    def __obtain_padding(self, input_shape: tuple)->tuple:
        """
        Function which obtains input shape of a tensor and returns values for padding
        Args:
            input_shape, tuple, obtainable by torch.shape 
        """

        _x, _y = input_shape
        _left_x  = (self.output_dim - _x) // 2
        _right_x = (self.output_dim - _x) - _left_x
        _left_y  = (self.output_dim - _y) // 2
        _right_y = (self.output_dim - _y)- _left_y
        return (_left_y, _right_y, _left_x, _right_x)

    def __interpolate_with_aspect_ratio(self, input_tensor: torch.Tensor, target_size: int = 224):
        """
        Interpolates a tensor while preserving aspect ratio.

        Args:
            input_tensor, torch.Tensor: Input tensor to be resized, with shape (N, C, H, W).
            target_size, int: Target size (height, width) for the resized tensor.

        Returns:
            torch.Tensor: Resized tensor with shape (N, C, target_size, target_size).
        """
        # Extract input size
        _input_size = input_tensor.size()[2:]
        _input_height, _input_width = _input_size

        # Calculate scale factors for resizing
        _scale_factor_height = target_size / _input_height
        _scale_factor_width = target_size / _input_width

        # Use the smaller scale factor to preserve aspect ratio
        _scale_factor = min(_scale_factor_height, _scale_factor_width)

        # Resize the tensor using bilinear interpolation
        _resized_tensor = F.interpolate(input_tensor, scale_factor=_scale_factor, mode='bilinear', align_corners=False)

        return _resized_tensor

class Shrink_Module_Naive_Shrinking(nn.Module):
    """
    Module which just calcualte the average sum of all pixels and pad it or reduce it to desired size. Paddng is done by zero pading, while
    reducing is done by bilinear interpolation
    
    """
    def __init__(self, output_dimension: int =  224):
        """
        Init functions
        Args:
            * output_dimension,int , output dimension after padding
        """
        super().__init__()
        
        # Define averaging or median layer
        self.coronal_shrink = MeanLayer(dim = 2)
        self.axial_shrink = MeanLayer(dim = 3)
        self.sagittal_shrink = MeanLayer(dim  = 4)

        # Define padding layers
        self.coronal_padding = PaddingLayer(output_dimension)
        self.axial_padding = PaddingLayer(output_dimension)
        self.sagittal_padding = PaddingLayer(output_dimension)    
        # Define cat layer
        self.concatenation_layer = CatLayer(dim = 1) 
        

    def forward(self, x):
        # Shrink to one channel by calculatin mean
        _x_coronal = self.coronal_shrink(x)
        _x_axial = self.axial_shrink(x)
        _x_sagittal = self.sagittal_shrink(x)
        
        # 0 pad 
        _x_coronal_paded = self.coronal_padding(_x_coronal)
        _x_axial_paded = self.axial_padding(_x_axial)
        _x_sagittal_paded = self.sagittal_padding(_x_sagittal)

        # Concat
        _x = self.concatenation_layer((_x_coronal_paded, _x_axial_paded, _x_sagittal_paded))
        return _x
    

