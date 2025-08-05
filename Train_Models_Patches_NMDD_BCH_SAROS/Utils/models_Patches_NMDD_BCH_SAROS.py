import torch
import torch.nn as nn
import torchinfo
import torchvision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchio as tio
from torch.nn import functional as F
import math

##############################################################
# Utils
##############################################################

def print_model_summary(model, 
                        device:str = 'cpu',
                        input_dim:tuple = (32, 1, 224, 224, 224)):
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
                                input_size)  # Define a sample input tensor
    # Run the sample input through the model
    model.eval()
    with torch.no_grad():
        _output = model(_sample_input)
    model.train()

    # Output
    _output = torch.flatten(_output)

    # Return value
    return _output.shape[0]

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
        
        for _param in model.mlp_head.parameters():
            _param.requires_grad = True
        return 
    
    if seq == 'vit' and freeze == False:
        for _param in model.parameters():
            _param.requires_grad = True
        
        for _param in model.mlp_head.parameters():
            _param.requires_grad = True
        
        return 

    #R3D_18 and MC3_18
    if seq == "18" and freeze == True:
        for _param in model.parameters():
            _param.requires_grad = False
        model.model.fc.weight.requires_grad = True
        model.model.fc.bias.requires_grad = True

    if seq == "18" and freeze == False:
        for _param in model.parameters():
            _param.requires_grad = True
        model.model.fc.weight.requires_grad = True
        model.model.fc.bias.requires_grad = True


    # Resnet
    if seq == True and freeze == True:
        for _param in model.features.parameters():
            _param.requires_grad = False
        
        for _param in model.initial_feature_block.parameters():
            _param.requires_grad = False

    if seq == True and freeze == False:
        for _param in model.features.parameters():
            _param.requires_grad = True
        
        for _param in model.initial_feature_block.parameters():
            _param.requires_grad = True


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
class VoxCNN_bulding_block(nn.Module):
    """
    Building block of the VoxCNN nn. It is composed of two 3D convoluitions and
    a Pool3D layer. Each convolution is followed by ReLU activation.
    Created to speed up the building process.
    """
    def __init__(self,  
                 number_of_input_channels = 1, 
                 number_of_kernels:int = 1,
                 extra: bool = False):
        """
        Init of the class:
        Args:
            * number_of_input_channels, int, number of input channels to the first 
            convolutional layer. Default value: 1 
            * number_of_kernels: int, number of kernels/channels in convolution layers.
            Default value: 1
            * extra, bool, adds extra convolution on the end
        """
        super().__init__()

        self.extra = extra
        # Build building blocks - convs
        self.first_conv_3D = nn.Conv3d(number_of_input_channels, 
                                      number_of_kernels,
                                      (3,3,3),
                                      stride = 1,
                                      padding = 1)
        self.second_conv_3D = nn.Conv3d(number_of_kernels, 
                                      number_of_kernels,
                                      (3,3,3),
                                      stride = 1,
                                      padding = 1)
        self.third_conv_3D = nn.Conv3d(number_of_kernels, 
                                      number_of_kernels,
                                      (3,3,3),
                                      stride = 1,
                                      padding = 1)
        
        # Activations
        self.first_ReLU = nn.ReLU()
        self.second_ReLU = nn.ReLU()
        self.third_ReLU = nn.ReLU()
        # Max pool
        self.maxPool3D = nn.MaxPool3d((2,2,2))


    def forward(self, x):
        """
        Forward pass
        """
        #_x = self.building_block(x)
        _x = self.first_conv_3D(x)
        _x = self.first_ReLU(_x)
        _x = self.second_conv_3D(_x)
        _x = self.second_ReLU(_x)
        if self.extra == True:
            _x = self.third_conv_3D(_x)
            _x = self.third_ReLU(_x)
        _x = self.maxPool3D(_x)
        return(_x)



class TransposeLayer(nn.Module):
    def __init__(self):
        super(TransposeLayer, self).__init__()
        
    def forward(self, x):
        # Transpose the tensor from (14, 4) to (4, 14)
        return x.transpose(0, 1)


class SimpleClassificationHead(nn.Module):
    """
    Simple classification head which only applies MaxPooling through the class dimension
    to obtain final prediction. 

    """

    def __init__(self, n_kernels:int = 27, max_pool = True):
        """
        Simple head which uses global max pooling
        """
        super().__init__()
        if max_pool:
            self.pool = nn.MaxPool1d(kernel_size = n_kernels) # Kernel size equal to the length of the sequence
        else:
            self.pool = nn.AvgPool1d(kernel_size = n_kernels) # Kernel size equal to the length of the sequence
        self.transpose_layer1 = TransposeLayer()
        self.transpose_layer2 = TransposeLayer()

    def forward(self, x):
        # Shape becomes (1, 14, 216*batch)
        _x = self.transpose_layer1(x)

        # Apply 1D max pooling
        _x_maxpooled = self.pool(_x)
        
        # Reshape back to (1, 14)
        _x_maxpooled = self.transpose_layer2(_x_maxpooled)

        return _x_maxpooled

class AttentioneBasedHead(nn.Module):
    """
    Implemented from stable diffusion repo:
    https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/attention.py
    """
    def __init__(self, n_heads, d_embed: int = 14, in_proj_bias=True, out_proj_bias=True):
        """
        Init function which creates all important adjustments. d_embed is embeding size, so in our case that should be 14
        which is number of output classes.
        """
        super().__init__()
         # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.pooling_layer = SimpleClassificationHead(max_pool = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, causal_mask=False):
        # x: # Seq_Len, Dim
        x = x.unsqueeze(0)
        # x: # (Batch_Size, Seq_Len, Dim)
        
        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        # (Batch_Size, Seq_Len, Dim)
        output = output.squeeze(0)
        
        # (Seq_Len, Dim)
        output = self.pooling_layer(output)

        # Apply sigmoid
        output = self.sigmoid(output)

        return output

class VoxCNN_Patches(nn.Module):
    """
    Model created based on the paper: DOI: 10.1109/ISBI.2017.7950647
    Sequential = False
    Channels = 1
    """

    def __init__(self,
                 input_shape:tuple = (1,110,110,110), 
                 output_classes:int = 1,
                 head:str = "Normal"):
        """
        Init of the class:
        Args:
            * input_shape, tupple, input shape of a voxel (channels, width, height, depth)
            * output_classes, int, integer representing number of output classes
            * head, str, "ATT" for attention, anything else for normal max pool.
        """
        super().__init__()

        # Create building blocks
        _number_of_kernels = [1,8,16,32,64]
        _features = []
        for _i in range(len(_number_of_kernels)-1):
            if _i >= len(_number_of_kernels)-3:
                _features += [VoxCNN_bulding_block(number_of_input_channels=_number_of_kernels[_i],
                                              number_of_kernels = _number_of_kernels[_i+1],
                                              extra = True)]
            else:
                _features += [VoxCNN_bulding_block(number_of_input_channels=_number_of_kernels[_i],
                                              number_of_kernels = _number_of_kernels[_i+1],
                                              extra = False)]
        # Build sequential part
        self.features = nn.Sequential(*_features)
        _output_shape = get_model_output_shape(model = self.features,
                                               input_channels = input_shape[0],
                                               input_size = (input_shape[1]))

        # Build head
        _flatten = nn.Flatten(start_dim = 1)
        _first_linear = nn.Linear(_output_shape, 128)
        _batch_norm = nn.BatchNorm1d(128)
        _dropout = nn.Dropout(p=0.7)
        _second_linear = nn.Linear(128, 64)
        _second_ReLU = nn.ReLU()
        _third_linear = nn.Linear(64, output_classes)
        _third_activation = nn.Sigmoid()
        self.classifier = nn.Sequential(*[_flatten,
                                          _first_linear,
                                          _batch_norm,
                                          _dropout,
                                          _second_linear,
                                          _second_ReLU,
                                          _third_linear,
                                          _third_activation])
        
        # init weights as in torchvision.models.video.resnet
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Add new head
        if head == 'ATT':
            self.classfication_head_patches = AttentioneBasedHead(n_heads=2)
        else:
            self.classfication_head_patches = SimpleClassificationHead()


    def forward(self, x):
        """
        Forward pass
        """
        _x = self.features(x)
        _x = self.classifier(_x)
        _x = self.classfication_head_patches(_x)
        return _x
    
##############################################################
##############################################################

class Norm(nn.Module):
    """
    Class which implements normalization
    """
    def __init__(self, dimension:int = 100):
        """
        Init of the class.
        Args:
            * dimension, int, just dimension for the normalisation
        """
        super().__init__()

        self.norm = nn.LayerNorm(dimension)

    def forward(self, x):
        """
        Forward pass
        """
        _x = self.norm(x)
        return _x
    
class Attention(nn.Module):
    """
    Class which implements attention module
    """

    def __init__(self,
                 io_dim:int = 2048,
                 number_of_heads: int = 8,
                 dim_head: int = 64,
                 dropout: int = 0.):
        """
        Init of the class.
        Args:
            * io_dim, int, dimension of the input/output of the attention module
            * number_of_heads, int, number of heads in every Multi head attention
            * dim_head, int, number of neurons in each head
            * dropout, int, size of the dropout
        """
        super().__init__()
        # Getting inner dimension for easier callculation
        _inner_dimension = number_of_heads * dim_head

        # Safety measure in cese that inner dimension is equal to output dim 
        # of the attention
        _projection = not (io_dim == _inner_dimension)

        # Set numbr of heads
        self.number_of_heads = number_of_heads

        # Calculate scale
        self.scale = dim_head ** -0.5

        # Attention
        self.attention = nn.Softmax(dim = -1)
        
        # Mapping layer -- clever way
        self.to_qkv = nn.Linear(io_dim, _inner_dimension * 3, bias = False)

        # Output
        if _projection == 1:
            self.output = nn.Sequential(
                nn.Linear(_inner_dimension, io_dim),
                nn.Dropout(dropout)
            )
        else:
            self.output = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass
        """
        # Calculateing key, querry and values at once
        _qkv = self.to_qkv(x).chunk(3, dim = -1)

        # Split them
        _q, _k, _v = map(lambda _t: rearrange(_t, 'b n (h d) -> b h n d', h = self.number_of_heads), _qkv)

        # Dot products
        _dot_products = torch.matmul(_q, _k.transpose(-1, -2)) * self.scale

        # Apply attention
        _attention = self.attention(_dot_products)

        # Calculate out_value
        _out = torch.matmul(_attention, _v)
        _out = rearrange(_out, 'b h n d -> b n (h d)')

        # Apply final output
        _out = self.output(_out)
        return _out

class FeedForward(nn.Module):
    """
    Class which is a simple ff nn
    """     
    def __init__(self,
                 io_dim, 
                 hidden_dim, 
                 dropout = 0.):
        """
        Init of the class
        Args:
            * io_dim, int, number of channels (dimension) of the vector
            * hidden_dim, int, number of channels (dimension) in the mlp hidden layer
            * droput, int, dropout coeffictient. Default: 0
        """   

        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(io_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, io_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass
        """
        _x = self.feedforward(x)
        return _x
    
class Transformer(nn.Module):
    """
    Class for building transformer
    """

    def __init__(self,
                 io_dim:int = 1000,
                 transformer_depth:int = 4,
                 number_of_heads:int = 16,
                 head_dim:int = 16,
                 mlp_dim:int = 100,
                 dropout = 0.):
        """
        Init of the class.
        Args:
            * io_dim, int, number of channels (dimension) of the input vector
            * transformer_depth, int, depth of transformer (how many stacks of blocks are there)
            * number_of_heads, int, number of heads in mha
            * transformer_depth, int, depth of the transformer (number of layers)
            * head_dim, int, number of neurons in each head
            * mlp_dim, int, number of hidden layers in mlp part of the block
            * droput, int, dropout coeffictient. Default: 0
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(transformer_depth):
            self.layers.append(nn.ModuleList([
                nn.Sequential(Norm(dimension = io_dim), 
                               Attention(io_dim = io_dim, 
                                         number_of_heads= number_of_heads, 
                                         dim_head = head_dim,
                                         dropout = dropout)
                ),
                nn.Sequential(Norm(dimension = io_dim),
                               FeedForward(io_dim = io_dim, 
                                           hidden_dim = mlp_dim, 
                                           dropout = dropout)

                )
            ]))

    def forward(self, x):
        """
        Forward pass
        """
        _x = x
        for _attention, _feed_forward in self.layers:
            _x = _attention(_x) + _x
            _x = _feed_forward(_x) + _x  
        return _x          

class ViT_3D(nn.Module):
    """
    Implementation based on https://www.kaggle.com/code/super13579/vit-vision-transformer-3d-with-one-mri-type
    Sequential = vit
    Channels = 1
    """

    def __init__(self, 
                 input_shape:tuple = (1,256),
                 patch_size: int = 32,
                 head_input_dim: int = 1024,
                 head_linear_dim: int = 2048,
                 number_of_classes: int = 14,
                 transformer_depth: int = 2,
                 number_of_heads:int = 16,
                 dropout: int = 0.1,
                 emb_dropout: int = 0.1):
        """
        Init of the class.
        Args:
            * input_shape, tupple, input shape of a voxel (channels, dimension_size)
            * patch_size, integer, size of a patch. Width, height and depth must be divisible with patch_size
            * head_input_dim, head_linear_dim, number_of_classes, int, number of channels in the mlp head 
            * transformer_depth, int, number of blocks in transformer (how many times will attention/ff be repeated)
            * number_of_heads, int, number of heads i each attention block (mha number)
            * droput, int, dropout coeffictient. Default: 0
        """

        super().__init__()
        
        # Calculate number of patches
        assert input_shape[1] % patch_size == 0, f"Input shape of a voxel must be divisiable by patch size. {input_shape}/{patch_size}"
        

        _number_of_patches = (input_shape[1] // patch_size) * (input_shape[1] // patch_size) * (input_shape[1] // patch_size)
        _squeezed_patch_size = input_shape[0] * patch_size ** 3
        self.patch_size = patch_size

        # Positional embeding
        self.pos_embedding = nn.Parameter(torch.randn(1, _number_of_patches + 1, head_input_dim))
        self.patch_to_embedding = nn.Linear(_squeezed_patch_size, head_input_dim)
        self.cls_token = nn.Parameter(torch.rand(1, 1, head_input_dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Build transformer
        self.transformer = Transformer(io_dim = head_input_dim,
                                       transformer_depth = transformer_depth,
                                       number_of_heads = number_of_heads,
                                       mlp_dim = head_linear_dim,
                                       dropout = dropout
                                       )
        
        # cls token SOMETHING
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 
                      head_linear_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_linear_dim, number_of_classes),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

   
    def forward(self, x, mask = None):
        """
        Forward pass
        """
        
        # Rearange
        _x = rearrange(x, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', 
                       p1 = self.patch_size, 
                       p2 = self.patch_size, 
                       p3 = self.patch_size)

        _x = self.patch_to_embedding(_x)

        _cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        _x = torch.cat((_cls_tokens, _x), dim = 1)

        _x += self.pos_embedding
        _x = self.dropout(_x)

        _x = self.transformer(_x)

        _x = self.to_cls_token(_x[:, 0])
        
        _x = self.mlp_head(_x)

        return _x
    
##############################################################

def split_tensor(input_tensor: torch.tensor, patch_size: tuple = (64,64,64), resize: bool = False)->torch.tensor:
    """
    Method which splits one tensor into smaller one.
    Args:
        * input_tensor, torch.tensor, input tensor whill will be splitted. Shape (1, 1, w, h, d)
        * patch_size, tuple(int,int,int), patch size of a subvoxel which is to be build
        * resize, boolean, if True the input_tensor will be resizied to closest volume divisable
        by patch_size in all dimensions.
    """
    # Resize to closses fully patchable dimension
    import torch.nn.functional as F
    if resize:
        #Calculate shape
        _new_shape = tuple((_dim + patch_size[0] // 2) // patch_size[0] * patch_size[0] for _dim in input_tensor.shape[2:])
        input_tensor = F.interpolate(input_tensor, size = _new_shape, mode='trilinear', align_corners=False)
    
    print(input_tensor.shape)
    # Define stride
    _stride = patch_size
    
    # Unfold
    _unfolded_tensor = input_tensor.unfold(2, patch_size[0], _stride[0]) \
                                   .unfold(3, patch_size[1], _stride[1]) \
                                   .unfold(4, patch_size[2], _stride[2])
    # Shape of unfolded tensor is [1, 1, num_block_x, num_blocks_y, num_blocks_z]
    _unfolded_tensor = _unfolded_tensor.contiguous().view(-1, _stride[0], _stride[1], _stride[2])

    # Display the shapes of the smaller tensors to verify
    print(f"Total number of smaller blocks: {_unfolded_tensor.shape[0]}")
    print(f"Shape of each smaller block: {_unfolded_tensor.shape[1:]}")

    # Print shapes of first few smaller tensors to verify
    for i in range(min(5, _unfolded_tensor.shape[0])):  # Display first 5 blocks
        print(f"Small tensor {i} shape: {_unfolded_tensor[i].shape}")

#*******************************************************#
# Augmentation models
#*******************************************************#  
class TransformToGray_SimplePatches:
    """
    Class which simply transform 1 chaneel to 3 channel gray image
    """
    def __init__(self, number_of_channels: int = 1, kernel_size = 128):
        """
        Simple transform which expand the model to the desired number of channels

        Args: 
            * number_of_channels, int, number of channels which the model expects as the input.
            typically 1 or 3
        """
        self.number_of_channels = number_of_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2

    def __call__(self, x):
        # Create 3 channel ima1e
        if self.number_of_channels == 1:
            _image = x.unsqueeze(1)
        if self.number_of_channels == 3:
            _image = x.unsqueeze(1)
            _image = _image.repeat(1, 3, 1, 1, 1)

        _x = _image.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        _x = _x.contiguous().view(-1, 1, self.kernel_size, self.kernel_size, self.kernel_size)

        return _x
    
class TransformToGray_AugmentedPatches:
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
        self.Add_Chanels = TransformToGray_SimplePatches( number_of_channels = number_of_channels)

    def __call__(self, x):
        # Unsqueeze
        #_image = torch.unsqueeze(x, 1)

        # Flip
        _image = self.filp_y(x)
        #_image = self.random_rotation(_image)
        #_image = self.gaussian_noise(_image)
        #_image = self.gaussian_blur(_image)
        
        # Remove channel
        #_image = _image.squeeze(1)

        # Add channels to make it 3 channeled image
        _image = self.Add_Chanels(_image)

        return _image
