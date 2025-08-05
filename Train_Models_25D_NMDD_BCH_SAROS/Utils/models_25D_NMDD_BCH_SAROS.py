import torch
import torch.nn as nn
import torchinfo
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import math
from Utils.shrinking_modules import *
import torchio as tio
from .shrinking_modules import *
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

def get_model_output_shape(model:nn.Module, 
                           input_channels:int = 1, 
                           input_size:int = 500)->int:
    """
    Get number of output neurons for a given model

    Args:
        * model, nn.module, model for which we are interested to find its output
        * input shape, int, input image with shape 
        (1, input_channels, input_size, input_size, input_size)
    """
    _sample_input = torch.randn(1, input_channels, 
                                input_size, 
                                input_size,
                                )  # Define a sample input tensor
    # Run the sample input through the model
    model.eval()
    with torch.no_grad():
        _output = model(_sample_input)
    model.train()

    # Output
    _output = torch.flatten(_output)

    # Return value
    return _output.shape[0]

def create_hook_model(model, model_type:str, verbose = False):
    """
    Function which returns new model which extract features from the orignial models.
    Features are the last conv layer (marked as "features") and pooling layers marked as
    "pooled".
    Args:
        * model, pytorch model
        * model_type, str, "vgg", "res", "eff"
        * verbose, bool, use to find model names
    
    Returns:
        * pytorch model
    """
    if verbose:
        train_nodes, eval_nodes = get_graph_node_names(model)
        print("Train:", train_nodes, "\n" "Eval:", eval_nodes)
        return 0

    # Check for types
    #assert model_type in ['vgg', 'res', 'eff', "incept_v3"], f"Model_type error. Supported types: vgg, res, eff"
    
    # Mobile nets
    if model_type == 'mobileNetV3Large':
        _return_nodes = {"features.16":"features", "avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model

    if model_type ==  'mobileNetV3Small':
        _return_nodes = {"features.12":"features", "avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model

    # ViTs
    if model_type in ['vit_b_16', 'vit_b_32', 'vit_l_16' , 'vit_l_32', 'vit_h_14']:
        _return_nodes = {"encoder.ln":"features"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model


    # AlexNet
    if model_type == 'alex':
        _return_nodes = {"features.12":"features", "avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model

    # VGG16
    if model_type == 'vgg16':
        _return_nodes = {"features.29":"features", "features.30":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
    
    # VGG19
    if model_type == 'vgg19':
        _return_nodes = {"features.36":"features", "avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
    
    #ResNet18
    if model_type == 'res18':
        _return_nodes = {"model.layer4.1.relu_1":"features", "model.avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
    

    #ResNet34
    if model_type == 'res34':
        _return_nodes = {"model.layer4.2.relu_1":"features", "model.avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model

    # ResNet50
    if model_type == 'res50':
        _return_nodes = {"model.layer4.2.relu_2":"features", "model.avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
    
    # ResNet101
    if model_type == 'res101':
        _return_nodes = {"model.layer4.2.relu_2":"features", "model.avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
    
    # ResNet152
    if model_type == 'res152':
        _return_nodes = {"model.layer4.2.relu_2":"features", "model.avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
    
    # EfficientNet
    if model_type in ['eff0', 'eff1', 'eff2', 'eff3', 'eff4', 'eef5', 'eff6', 'eff7']:
        _return_nodes = {"features.8":"features", "avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
    
    if model_type == 'incept_v3':   
        _return_nodes = {"model.Mixed_7c.cat_2":"features", "model.avgpool":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
        
    # DenseNet121
    if model_type in ['denseNet121', 'denseNet161']:
        _return_nodes = {"relu":"features", "adaptive_avg_pool2d":"pooled"}
        _intermediate_model = create_feature_extractor(model, return_nodes=_return_nodes)
        return _intermediate_model
            
def freeze_model_part(model, freeze_ratio:float):
    """
    Method which freezes part of the model.
    
    Args:
        * model, pytorch model, model which is suppose to be frozen
        * freeze_ratio, float, part of the model which is going to be frozen. 0.8 means that first 
        80% of the layers will be frozen.
    """

    print(f"Freezing ratio: {freeze_ratio}")


    # First bring everything trainable - starting position
    for _param in model.parameters():
            _param.requires_grad = True
            
    # Calculate ratio
    _number_of_layers = len(list(model.named_parameters()))
    _freeze_border = int(freeze_ratio * _number_of_layers)
    
    # Freeze layer
    for _i, _param in enumerate(model.parameters()):
        if _i < _freeze_border:
            _param.requires_grad = False
        
    # Fix bias layer - params + bias must both be frozen
    for _name, _param in model.named_parameters():
        if _param.requires_grad and 'bias' in _name:
            _param.requres_grad = False

def freeze_model_base(model, freeze:bool = True, seq = False):
    """
    Script which (un)freezes model's base paramters 

    Args:
        * model, pytorch model
        * freeze, boolean, if True parameters are Frozen, False unfreezes 
        * seq, bool, for some models necessary model then it needs to be set true

    """
    #vit
    if seq == 'vit' and freeze == True:
        for _param in model.parameters():
            _param.requires_grad = False
        model.heads.head.weight.requires_grad = True
        model.heads.head.bias.requires_grad = True
        return 
    
    if seq == 'vit' and freeze == False:
        for _param in model.parameters():
            _param.requires_grad = True
        return 


    # Resnet
    if seq and freeze == True:
        for _param in model.parameters():
            _param.requires_grad = False
        model.model.fc.weight.requires_grad = True
        model.model.fc.bias.requires_grad = True
        return 
    
    if seq and freeze == False:
        for _param in model.parameters():
            _param.requires_grad = True
        return  

    # Efficientnet and VGG
    if seq == False and freeze == True:
        for _param in model.features.parameters():
            _param.requires_grad = False

    if seq == False and freeze == False:
        for _param in model.features.parameters():
            _param.requires_grad = True

##############################################################
# Models
##############################################################

class mobileNetV3Small(nn.Module):
    """
    Class for building ResNet34 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(mobileNetV3Small, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.mobilenet_v3_small(weights=_weights).to(device)
        
        self.shrinking_module = Shrink_Module_1x1Conv2D(in_channels = 224)
        self.features = _model.features
        self.avgpool = _model.avgpool
        self.classifier = _model.classifier
        self.classifier[-1] = nn.Linear(1024, number_of_classes)

    def forward(self, x):
        _x = self.shrinking_module(x) 
        _x = self.features(_x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class mobileNetV3Large(nn.Module):
    """
    Class for building ResNet34 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(mobileNetV3Large, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.mobilenet_v3_large(weights=_weights).to(device)
        
        self.features = _model.features
        self.avgpool = _model.avgpool
        self.classifier = _model.classifier
        self.classifier[-1] = nn.Linear(1280, number_of_classes)

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class vit_l_16(nn.Module):
    """
    Class for building vit
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet 224x224

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(vit_l_16, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.ViT_L_16_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.vit_l_16(weights=_weights).to(device)
        
        # Necessary things for models
        self.patch_size = _model.patch_size
        self.conv_proj = _model.conv_proj
        self.image_size = _model.image_size
        self.hidden_dim = _model.hidden_dim
        self.class_token = _model.class_token

        # Encoder
        self.encoder = _model.encoder

        # Heads
        _model.heads.head = nn.Linear(_model.heads.head.in_features, number_of_classes)
        self.heads = _model.heads


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function for creating tokens
        """
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

class vit_l_32(nn.Module):
    """
    Class for building vit
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(vit_l_32, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.ViT_L_32_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.vit_l_32(weights=_weights).to(device)
        
        # Necessary things for models
        self.patch_size = _model.patch_size
        self.conv_proj = _model.conv_proj
        self.image_size = _model.image_size
        self.hidden_dim = _model.hidden_dim
        self.class_token = _model.class_token

        # Encoder
        self.encoder = _model.encoder

        # Heads
        _model.heads.head = nn.Linear(_model.heads.head.in_features, number_of_classes)
        self.heads = _model.heads


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function for creating tokens
        """
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

class vit_h_14(nn.Module):
    """
    Class for building vit
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet 518x1518

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(vit_h_14, self).__init__()

        # Load model pretrained on vit_h_14 imagenet
        if pretrained: 
            _weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.vit_h_14(weights=_weights).to(device)
        
        # Necessary things for models
        self.patch_size = _model.patch_size
        self.conv_proj = _model.conv_proj
        self.image_size = _model.image_size
        self.hidden_dim = _model.hidden_dim
        self.class_token = _model.class_token

        # Encoder
        self.encoder = _model.encoder

        # Heads
        _model.heads.head = nn.Linear(_model.heads.head.in_features, number_of_classes)
        self.heads = _model.heads

        # Final sigmoid
        self.sigmoid = nn.Sigmoid()

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function for creating tokens
        """
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        x = self.sigmoid(x)
        return x

class vit_b_16(nn.Module):
    """
    Class for building vit
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(vit_b_16, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.ViT_B_16_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.vit_b_16(weights=_weights).to(device)
        
        # Necessary things for models
        self.patch_size = _model.patch_size
        self.conv_proj = _model.conv_proj
        self.image_size = _model.image_size
        self.hidden_dim = _model.hidden_dim
        self.class_token = _model.class_token

        # Encoder
        self.encoder = _model.encoder

        # Heads
        _model.heads.head = nn.Linear(_model.heads.head.in_features, number_of_classes)
        self.heads = _model.heads


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function for creating tokens
        """
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
         
class vit_b_32(nn.Module):
    """
    Class for building vit
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(vit_b_32, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.ViT_B_32_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.vit_b_32(weights=_weights).to(device)
        
        # Necessary things for models
        self.patch_size = _model.patch_size
        self.conv_proj = _model.conv_proj
        self.image_size = _model.image_size
        self.hidden_dim = _model.hidden_dim
        self.class_token = _model.class_token
        print(self.patch_size)
        # Encoder
        self.encoder = _model.encoder

        # Change heads
        _model.heads.head = nn.Linear(_model.heads.head.in_features, number_of_classes)
        self.heads = _model.heads


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function for creating tokens
        """
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

class DenseNet121(nn.Module):
    """
    Class for building densnet
    https://pytorch.org/vision/main/_modules/torchvision/models/densenet.html
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(DenseNet121, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.DenseNet121_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.densenet121(weights=_weights).to(device)
        
        # Get Features
        self.features = _model.features
        # Edit LastLayer
        _model.classifier = nn.Linear(_model.classifier.in_features, number_of_classes)
        self.classifier = _model.classifier    
        
        # Transfer
        self.model = _model
            
    def forward(self, x):
        _features = self.features(x)
        _out = F.relu(_features, inplace=True)
        _out = F.adaptive_avg_pool2d(_out, (1, 1))
        _out = torch.flatten(_out, 1)
        _x = self.classifier(_out)
        return _x
           
class DenseNet161(nn.Module):
    """
    Class for building densnet
    https://pytorch.org/vision/main/_modules/torchvision/models/densenet.html
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(DenseNet161, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.DenseNet161_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.densenet161(weights=_weights).to(device)
        
        # Get Features
        self.features = _model.features
        # Edit LastLayer
        _model.classifier = nn.Linear(_model.classifier.in_features, number_of_classes)
        self.classifier = _model.classifier    
        
        # Transfer
        self.model = _model

        # Final sigmoid
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, x):
        _features = self.features(x)
        _out = F.relu(_features, inplace=True)
        _out = F.adaptive_avg_pool2d(_out, (1, 1))
        _out = torch.flatten(_out, 1)
        _x = self.classifier(_out)
        _x = self.sigmoid(_x)
        return _x

class InceptionV3(nn.Module):
    """
    Class for building InceptionV3 neural network
    """ 

    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """

        # Inherent
        super(InceptionV3, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.inception.Inception_V3_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.inception_v3(weights=_weights).to(device)

        # Edit LastLayer
        _model.fc = nn.Linear(2048, number_of_classes)
        
        self.model = _model


    def forward(self, x):
        _x = self.model(x)

        return _x

class AlexNet(nn.Module):
    """
    Class for building InceptionV3 neural network
    """ 

    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """

        # Inherent
        super(AlexNet, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.inception.AlexNet_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.alexnet(weights=_weights).to(device)

        # Edit features extractors
                # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool

        self.classifier = _model.classifier
        self.classifier[-1] = nn.Linear(4096, number_of_classes)

    def forward(self, x):
        _x = self.features(x)
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class VGG19(nn.Module):
    """
    Class for building VGG16 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(VGG19, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained:
            _weights = torchvision.models.VGG19_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.vgg19(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        self.classifier = _model.classifier
        self.classifier[-1] = nn.Linear(4096, number_of_classes)

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class VGG16(nn.Module):
    """
    Class for building VGG16 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(VGG16, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained:
            _weights = torchvision.models.VGG16_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.vgg16(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, number_of_classes)
        )

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        #_x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class EfficientNetB1(nn.Module):
    """
    Class for building EfficeientNetB1 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 240x240

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB1, self).__init__()

        # Load model pretrained on EfficientNet_B1 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b1(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(1280, number_of_classes) #153600 for ff
        )

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class EfficientNetB2(nn.Module):
    """
    Class for building EfficeientNetB2 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 260x260


        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB2, self).__init__()

        # Load model pretrained on EfficientNet_B2 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b2(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(1408, number_of_classes) #153600 for ff
        )

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class EfficientNetB3(nn.Module):
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 300x300

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB3, self).__init__()

        # Load model pretrained on EfficientNet_B3 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b3(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(1536, number_of_classes) #153600 for ff
        )

        # Init classifier
        if pretrained == False:
            self.classifier.apply(init_weights)

        # Final sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        _x = self.sigmoid(_x)
        return _x
    
class EfficientNetB4(nn.Module):
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 380x380

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB4, self).__init__()

        # Load model pretrained on EfficientNet_B3 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b4(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(1792, number_of_classes) #153600 for ff
        )

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x
    
class EfficientNetB5(nn.Module):
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 456x456

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB5, self).__init__()

        # Load model pretrained on EfficientNet_B3 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b5(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(2048, number_of_classes) #153600 for ff
        )

        # Final sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        _x = self.sigmoid(_x)
        return _x
    
class EfficientNetB6(nn.Module):
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 528x528

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB6, self).__init__()

        # Load model pretrained on EfficientNet_B3 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b6(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(2304, number_of_classes) #153600 for ff
        )

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class EfficientNetB7(nn.Module):
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 600x600

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB7, self).__init__()

        # Load model pretrained on EfficientNet_B3 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b7(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(2560, number_of_classes) #153600 for ff
        )

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

class ResNet18(nn.Module):
    """
    Class for building ResNet18 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet18, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet18_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet18(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(512, number_of_classes)

        # Build model
        self.model = _model

    def forward(self, x):
        #_x = self.features(x)
        # For any input image size
        
        _x = self.model(x)
        return _x

class ResNet34(nn.Module):
    """
    Class for building ResNet34 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet34, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet34_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet34(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(512, number_of_classes)

        # Build model
        self.model = _model

    def forward(self, x):
        #_x = self.features(x)
        # For any input image size
        
        _x = self.model(x)
        return _x
   
class ResNet101(nn.Module):
    """
    Class for building ResNet50 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet101, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet101_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet101(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(2048, number_of_classes)

        # Build model
        self.model = _model

    def forward(self, x):
        #_x = self.features(x)
        # For any input image size
        
        _x = self.model(x)
        return _x
    
class ResNet152(nn.Module):
    """
    Class for building ResNet50 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet152, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet152_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet152(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(2048, number_of_classes)

        # Build model
        self.model = _model

    def forward(self, x):
        #_x = self.features(x)
        # For any input image size
        
        _x = self.model(x)
        return _x
    
class ResNet50(nn.Module):
    """
    Class for building ResNet50 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet50, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet50_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet50(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(2048, number_of_classes)

        # Build model
        self.model = _model

        # Shrink module
        self.shrink_module = Shrink_Module_1x1Conv2D()

        # Output final
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        _x = self.shrink_module(x) 
        _x = self.model(_x)
        _x = self.final_activation(_x)
        return _x

class EfficientNetB0(nn.Module): 
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet
        Input size: 224x224

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB0, self).__init__()

        # Load model pretrained on EfficientNet_B0 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b0(weights=_weights).to(device)
        
        # Init random weights of the model if the model does not have pretrained weigths
        if pretrained == False:
            _model.apply(init_weights)

        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(1280, number_of_classes) #153600 for ff
        )
        # Init classifier
        if pretrained == False:
            self.classifier.apply(init_weights)

        # Shrink module
        self.shrink_module = Shrink_Module_1x1Conv2D()

        # Output final
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        _x = self.shrink_module(x)
        _x = self.features(_x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        _x = self.final_activation(_x)
        return _x
    
def init_weights(module):
    """
    Method whcih inits random weights to layers.
    """
    _init_set = {
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
    }

    for _m in module.modules():
        if type(_m) in _init_set:
            nn.init.kaiming_normal_(
                _m.weight.data, mode='fan_out', nonlinearity='relu', a=0
            )
            if _m.bias is not None:
                fan_in, fan_out = \
                    nn.init._calculate_fan_in_and_fan_out(_m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(_m.bias, -bound, bound)


class DenseNet161(nn.Module):
    """
    Class for building densnet
    https://pytorch.org/vision/main/_modules/torchvision/models/densenet.html
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(DenseNet161, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.DenseNet161_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.densenet161(weights=_weights).to(device)
        
        # Get Features
        self.features = _model.features
        # Edit LastLayer
        _model.classifier = nn.Linear(_model.classifier.in_features, number_of_classes)
        self.classifier = _model.classifier    
        
        # Transfer
        self.model = _model

        # Shrink module
        self.shrink_module = Shrink_Module_1x1Conv2D()

        # Output final
        self.final_activation = nn.Sigmoid()
            
    def forward(self, x):
        _x = self.shrink_module(x)
        _features = self.features(_x)
        _out = F.relu(_features, inplace=True)
        _out = F.adaptive_avg_pool2d(_out, (1, 1))
        _out = torch.flatten(_out, 1)
        _x = self.classifier(_out)
        _x = self.final_activation(_x)
        return _x



#*******************************************************#
# Augmentation models
#*******************************************************#  
class TransformToGray_Simple:
    """
    Class which simply transform 1 chaneel to 3 channel gray image
    """
    def __init__(self, number_of_channels: int = 1):
        """
        Simple transform which expand the model to the desired number of channels

        Args: 
            * number_of_channels, int, number of channels which the model expects as the input.
            typically 1 or 3
        """
        self.number_of_channels = number_of_channels
    
    def __call__(self, x):
        # Create 3 channel ima1e
        #if self.number_of_channels == 1:
        #    _image = x.unsqueeze(1)
        #if self.number_of_channels == 3:
        #    _image = x.unsqueeze(1)
        #    _image = _image.repeat(1, 3, 1, 1, 1)
        return x



class TransformToGray_Augmented:
    """
    https://www.frontiersin.org/articles/10.3389/fmed.2021.629134/full
    https://www.imaios.com/en/resources/blog/ai-for-medical-imaging-data-augmentation
    https://torchio.readthedocs.io/transforms/augmentation.html#randomnoise
    Class which transform grayscale data by applying:

        * Flipping by the Y axis (reflection): probabiltiy 0.5
        * Random rotation for +- 15 degrees
        * Random Gaussian noise (mu = 0.05, std = 0.05)
        * Random Blur (std = 0.1, 5.0) 
    """
    def __init__(self, number_of_channels: int = 1):
        """
        Define all transofrmations
        
        Args: * number_of_channels, int, number of channels which the model expects as the input - typically 1 or 3
        """
        self.filp_y = tio.transforms.RandomFlip(axes=(1,), p=0.5)
        self.random_rotation = tio.transforms.RandomAffine(scales=(1, 1), degrees=(0, 0, 15))
        self.gaussian_noise = tio.transforms.RandomNoise(mean=0.05, std=0.05)
        self.gaussian_blur = tio.transforms.RandomBlur(std=(0.1, 1.0))
        
        # Finally add channels
        #self.Add_Chanels = TransformToGray_Simple( number_of_channels = number_of_channels)

    def __call__(self, x):
        # Unsqueeze
        #_image = torch.unsqueeze(x, 1)

        # Flip
        _image = self.filp_y(x)
        #_image = self.random_rotation(_image)
        _image = self.gaussian_noise(_image)
        #_image = self.gaussian_blur(_image)
        
        # Remove channel
        #_image = _image.squeeze(1)

        # Add channels to make it 3 channeled image
        #_image = self.Add_Chanels(_image)

        return _image