from collections import namedtuple
import torch
from Utils.models_3D_NMDD_BCH_SAROS import *
import numpy as np
import datetime
import pandas as pd
import xlsxwriter
from sklearn import metrics
import time
import shutil
import os
import wandb
from Utils.dataloader_3D_NMDD_BCH_SAROS import *
from Utils.config_3D_NMDD_BCH_SAROS import *
from torch.optim import lr_scheduler

class model_training_app:
    def __init__(self, 
                 train_dl = None, 
                 valid_dl = None, 
                 model_params = None, 
                 results_output_dir = None):
        """
        init training with given params
        
        Args:
            * train_dl, train data set dataloader
            * valid_dl, validation data set dataloader
            * model_params, name+model params. Names: "vgg, eff, res"
            * results_output_dir, str, path to output dir for results
        """
        print("**************************************")

        # Define output dir (delete if exists + create new one)
        self.results_output_dir = results_output_dir
        if os.path.exists(self.results_output_dir):
            shutil.rmtree(self.results_output_dir)
        os.makedirs(self.results_output_dir)

        # Load data
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model_params = model_params

        # Set device
        if self.model_params.gpu == False or self.model_params.gpu == 'cpu' :
            self.device = torch.device("cpu")
        else:
            self.use_cuda = torch.cuda.is_available()        
            self.device = torch.device(self.model_params.gpu)
        
        # Get model, Optimizer and augumentation model
        self.model = self.init_model()

        # Augumentation model
        self.aug_model = None

        if self.model_params.augmentation_model != None:
            self.init_augmentation_model()

        # Loss function
        self.loss = self.init_loss()
        
        # Init optimzer (must be last for lr find)
        self.optimizer = self.init_optimizer()

        # Wandb
        if self.model_params.wandb == True:
            self.init_wandb()

        print("**************************************")


    #*******************************************************#
    # Model handling scripts
    #*******************************************************#   
    def init_model(self):
        """
        Model is initially unfrozen
        """
        assert self.model_params.model_name in ['VoxCNN', 'ResNet', 'R3D_18', 'MC3_18', 'R2plus1d_18', 'ViT_3D'], f"Model not implemented!"

        print(f"USING MODEL: {self.model_params.model_name}, WEIGHTS: UNFROZEN")

        # Currently implemented 
        if self.model_params.model_name == 'VoxCNN':
            _model = VoxCNN(input_shape = self.model_params.input_shape['input_shape'], output_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)
            
        if self.model_params.model_name == 'ResNet':
            _model = ResNet(input_shape = self.model_params.input_shape['input_shape'], output_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = True)

        if self.model_params.model_name == 'R3D_18':
            _model = R3D_18(pretrained = self.model_params.input_shape["pretrained"], output_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = "18")

        if self.model_params.model_name == 'MC3_18':
            _model = MC3_18(pretrained = self.model_params.input_shape["pretrained"], output_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = "18")

        if self.model_params.model_name == 'R2plus1d_18':
            _model = R2plus1d_18(pretrained = self.model_params.input_shape["pretrained"], output_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = "18")

        if self.model_params.model_name == 'ViT_3D':
            _model = ViT_3D(input_shape = self.model_params.input_shape["input_shape"],
                            patch_size = self.model_params.input_shape["patch_size"],
                            head_input_dim = self.model_params.input_shape["head_input_dim"],
                            head_linear_dim = self.model_params.input_shape["head_linear_dim"],
                            number_of_classes = self.model_params.input_shape["number_of_classes"],
                            transformer_depth = self.model_params.input_shape["transformer_depth"],
                            number_of_heads = self.model_params.input_shape["number_of_heads"],
                            dropout = self.model_params.input_shape["dropout"],
                            emb_dropout = self.model_params.input_shape["emb_dropout"])
            freeze_model_base(_model, freeze = False, seq = "vit")

        # Send it to gpu
        if self.model_params.gpu != False and self.model_params.gpu != "cpu":
            print(f"USING GPU: {self.device}")
            _model = _model.to(self.device)
        else:
            print("USING CPU")

        return _model        

    def freeze_unfreeze_model(self, freeze: bool = True):
        """
        Function which freezes and unfreezes models.

        Args:
            * model, pytorch model
            * freeze, bool, True for freeze and False for unfreeze
        """
        # Set seq
        if self.model_params.model_name in ['VoxCNN']:
            _seq = False
        if self.model_params.model_name in ['ResNet']:
            _seq = True
        if self.model_params.model_name in ['R3D_18', 'MC3_18', 'R2plus1d_18']:
            _seq = "18"
        if self.model_params.model_name in ['ViT_3D']:
            _seq = 'vit'

        # Freeze or unfreeze
        freeze_model_base(self.model, freeze = freeze, seq = _seq)

        # Notice
        if freeze == True:
            print(f"USING MODEL: {self.model_params.model_name}, WEIGHTS: FROZEN")
        else:
            print(f"USING MODEL: {self.model_params.model_name}, WEIGHTS: UNFROZEN")

        # Refresh optimizer settings
        self.optimizer = self.init_optimizer()    

        # Send it to gpu  
        if self.model_params.gpu != False and self.model_params.gpu != 'cpu':
            print(f"USING GPU: {self.device}")
            self.model = self.model.to(self.device)
        else:
            print("USING CPU")

        print("**************************************")

    def load_model(self, path):
        """
            Function that loads model.

            Args:
                * path, string, path to the model checkpoint
        """
        print("LOADING MODEL")
        
        _state_dict = torch.load(path, map_location = self.device)
        self.model.load_state_dict(_state_dict['model_state'])
        self.optimizer.load_state_dict(_state_dict['optimizer_state'])
        self.optimizer.name = _state_dict['optimizer_name']
        self.model.name = _state_dict['optimizer_name']
        
        print(f"LOADING MODEL, epoch {_state_dict['epoch']}"
                 + f", time {_state_dict['time']}")
    
    def save_model(self, epoch, best, info):
        """
            Function for model saving

            Args:
                * epoch, int, epoch being saved
                * best, boolean, Is this the best model
                * info, str, decoration for the name
        """

        _name = f"{self.model_params.model_name}_{self.model_params.opt_name}:" +  \
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}" + \
                    f"_{epoch}"

        _model = self.model
        
        # For paralel
        if isinstance(_model, torch.nn.DataParallel):
            _model = _model.module
        
        # Define saving state
        _state = {
            'time': str(datetime.datetime.now()),
            'model_state': _model.state_dict(),
            'model_name': type(_model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }
        
        # Save last model
        torch.save(_state, self.results_output_dir+_name + '.pth')
        print('Saving model!')
        
        # Save best model
        if best:
            print('Saving best model!')
            _name = f"{self.model_params.model_name}_{self.model_params.opt_name}_" +  \
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}" 
            torch.save(_state, self.results_output_dir + _name + f'{info}_best_model.pth')
    
    #*******************************************************#
    # Init augumentation model
    #*******************************************************# 
    def init_augmentation_model(self):
        """
        Script which inits augumentation model
        """
        if self.model_params.augmentation_model == 'GRAY_Augmentation':
            self.aug_model = TransformToGray_Augmented(number_of_channels = self.model_params.input_channels)
            self.aug_model_color_only = TransformToGray_Simple(number_of_channels = self.model_params.input_channels)
                        
        if self.model_params.augmentation_model == 'GRAY_Simple':
            self.aug_model = TransformToGray_Simple(number_of_channels = self.model_params.input_channels)
            self.aug_model_color_only = TransformToGray_Simple(number_of_channels = self.model_params.input_channels)

    #*******************************************************#
    # Loss function
    #*******************************************************#
    def init_loss(self):
        """
        Init loss function: Feel free to add other loss functions.
        """
        assert self.model_params.loss_name in ['bce'], f"Loss not implemented!"
        
        print(f"USING LOSS FUNCTION: {self.model_params.loss_name}")
        if self.model_params.loss_name == "bce":
            return torch.nn.BCELoss(reduction='none')
        
    
    #*******************************************************#
    # Optimzer
    #*******************************************************#  
    def init_optimizer(self):
        """
            Init optimizer: Feel free to add other optmizers. 
        """     
        
        assert self.model_params.opt_name in ["ADAM", "ADAMW", "SGD"], f"Wrong optimizer name, got: {self.model_params.opt_name}"
        print(f"USING OPTIMIZER: {self.model_params.opt_name} / LR:{self.model_params.learning_rate}")

        # Set optimizer
        if self.model_params.opt_name == 'ADAM':
            return torch.optim.Adam(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
        if self.model_params.opt_name == 'ADAMW':
            return torch.optim.AdamW(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
        if self.model_params.opt_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(),  lr = self.model_params.learning_rate)
        
    #*******************************************************#
    # Wandb init script
    #*******************************************************# 
    def init_wandb(self, time_stamp = None, name = "Training"):
        """
        Init function for wandb

        Args:
            * name, name of the training
        """
        # Obtain time
        if time_stamp == None:
            _current_time = time.strftime("%H_%M_%S", time.localtime())
        else:
            _current_time = time_stamp

        # Obtain augumentation
        if self.model_params.augmentation_model != None:
            _aug = True
        else:
            _aug = False

        wandb.init(
        # set the wandb project where this run will be logged
        project=f"{self.model_params.model_name}_"+f"{_current_time}",
        name = f"{name}",
        # track hyperparameters and run metadata
        config={f'name': self.model_params.model_name,
              f'info': self.model_params.info,  
              f'epochs': self.model_params.epochs,
              f'valid_epochs': self.model_params.valid_epochs, 
              f'early_stopping': self.model_params.early_stopping, 
              f'opt_name': self.model_params.opt_name,
              f'learning_rate': self.model_params.learning_rate, 
              f'loss_name': self.model_params.loss_name, 
              f'augumentation_model': _aug,
              f'gpu': self.model_params.gpu
            }
        )

    #*******************************************************#
    # Training subrutine
    #*******************************************************#
    def train_one_epoch(self, data):
        """
        Training model function. 

        Args:
            * data, dataloader of the train dataset
        """

        # Storage for metrics calculation
        _predictions_storage = torch.zeros(len(data.dataset) * self.model_params.number_of_output_classes, device = self.device)
        _true_storage = torch.zeros(len(data.dataset) * self.model_params.number_of_output_classes, device = self.device)
        _loss_storage = torch.zeros(len(data), device = self.device)
        
        # Swap to mode train
        self.model.train()

        # Shuffle dataset and create enum object
        data.dataset.shuffle_samples()
        _batch_iter = enumerate(data)
        
        # Go trough batches
        for _index, _batch in _batch_iter:
            # Clear grads
            self.optimizer.zero_grad()

            # Calc loss
            _loss = self.get_loss(index = _index, 
                                  batch = _batch, 
                                  predictions_storage = _predictions_storage, 
                                  true_storage = _true_storage, 
                                  augmentation = True)
                
            # Propagate loss
            _loss.backward()

            # Apply loss
            self.optimizer.step()

            # Save loss
            _loss_storage[_index] = _loss.detach()
        
        # Return metrics
        return _predictions_storage, _true_storage, _loss_storage

    
    #*******************************************************#
    # Validation subrutine
    #*******************************************************#
    def validate_model(self, data):
        """
        Validation model function

        Args:
            * data, dataloader of the train dataset
        """

        # # Storage for metrics calculation
        # Storage for metrics calculation
        _predictions_storage = torch.zeros(len(data.dataset) * self.model_params.number_of_output_classes, device = self.device)
        _true_storage = torch.zeros(len(data.dataset) * self.model_params.number_of_output_classes, device = self.device)
        _loss_storage = torch.zeros(len(data), device = self.device)

        # We don't need calculate gradients 
        with torch.no_grad():
            # Set model in evaluate mode - no batchnorm and dropout
            self.model.eval()

            # Go trough data
            for _index, _batch in enumerate(data):
                # Get loss
                _loss = self.get_loss(index = _index, 
                                                batch = _batch, 
                                                predictions_storage = _predictions_storage, 
                                                true_storage = _true_storage, 
                                                augmentation = False)
                
                # Save loss
                _loss_storage[_index] = _loss.detach()
        
        # Return staff
        return _predictions_storage, _true_storage, _loss_storage

    #*******************************************************#
    # BCE loss
    #*******************************************************#
    def get_loss(self, index, batch, predictions_storage, true_storage, augmentation = True):
        """
        Function that calculates cross entropy loss. Loss in this code is MeanSquaredError

        Args:
            * index, int, batch index needed to populate _metrics

            * batch, tensor, data

            * predictions_storage, true_storage, lists, lists to store predictions and true values of batch

            * augmentation, boolean, True if augumentation is to be applied
        """
        # Parse _batch
        _image, _label, _, _ = batch
        # Transfer data to GPU
        _input_data = _image.to(self.device, non_blocking = True)
        _input_data = _image
        _output_data = _label.to(self.device, non_blocking = True)
        
        # Augment data
        if self.aug_model != None and augmentation == True:
            _input_data = self.aug_model(_input_data)
            
        # If augumentation is required only for color scheme (validation set)
        if self.aug_model != None and augmentation == False:
            _input_data = self.aug_model_color_only(_input_data)
        
        _input_data = _input_data.to(self.device, non_blocking = True)
        # Caluclate loss
       	_prediction = self.model(_input_data)	

        _loss = self.loss(_prediction, _output_data)

        # Detach from graph
        _prediction = _prediction.detach()
        _output_data = _output_data.detach()

        # For metrics
        with torch.no_grad():
            _prediction = _prediction.flatten()
            _output_data = _output_data.flatten()
            # Fix last batch size
            predictions_storage[index * self.train_dl.batch_size * self.model_params.number_of_output_classes: index * self.train_dl.batch_size * self.model_params.number_of_output_classes + _prediction.shape[0] ] = _prediction
            true_storage[index * self.train_dl.batch_size * self.model_params.number_of_output_classes: index * self.train_dl.batch_size * self.model_params.number_of_output_classes + _output_data.shape[0] ] = _output_data
        # Return mean of all loss          
        return _loss.mean()

    #*******************************************************#
    # Function for evaluation, 
    #*******************************************************#
    def eval_metrics(self, epoch, predictions, true, loss, mode ,save_dict)->float:
        """
            Function for metric evaluation

        Args:
            * epoch, int, epoch number
            * predictions, torch.tensor, prediction for each batch
            * true, torch.tensor, true value for each batch
            * loss, torch.tensor, loss values
            * mode, str, 'valid', 'train', 'test' - just cosmetic
            * save_dict, dict, dictionary for metrics update

        Return:
            * Calculated loss over complete dataset
        
        """
        # Transfer to cpu
        _predictions = predictions.to('cpu')
        _true = true.to('cpu')

        # Get 1D array
        _predictions = torch.flatten(_predictions)
        _true = torch.flatten(_true)
        
        # Thershold 
        _predictions = torch.where(_predictions > 0.5, torch.tensor(1.0), torch.tensor(0.0))

        # Calculate precission recall f1 score
        _report = metrics.classification_report(_predictions, _true, digits=3, zero_division = 0, output_dict = True)
        _weighted_report = _report['weighted avg']
        _macro_report = _report['macro avg']
 
        # Print info
        print("{}, {}, Loss:{:.3f}".format(mode, epoch, torch.mean(loss.double())))
        print("{}, {}, Weighted : Precision:{:.3f}, Recall:{:.3f} F1-score: {:.3f}".format(mode, epoch, 
                                                                                 _weighted_report['precision'],
                                                                                 _weighted_report['recall'],
                                                                                 _weighted_report['f1-score']))
        print("{}, {}, Macro : Precision:{:.3f}, Recall:{:.3f} F1-score: {:.3f}".format(mode, epoch, 
                                                                                 _macro_report['precision'],
                                                                                 _macro_report['recall'],
                                                                                 _macro_report['f1-score']))
        
        # Save to dict
        save_dict['epoch'].append(epoch)
        save_dict['loss'].append(torch.mean(loss.double()))
        save_dict['precision'].append(_macro_report['precision'])
        save_dict['recall'].append(_macro_report['recall'])
        save_dict['f1-score'].append(_macro_report['f1-score'])
        save_dict['precision-w'].append(_weighted_report['precision'])
        save_dict['recall-w'].append(_weighted_report['recall'])
        save_dict['f1-score-w'].append(_weighted_report['f1-score'])

                
        # Return loss value for model evaluation, or any other score.
        return torch.mean(loss.double())
        
    #*******************************************************#
    # Export metrics to xlsx
    #*******************************************************#
    def export_metrics_to_xlsx(self, best_epoch, best_score, training_dict, validation_dict):
        """
        Function that exports model's training and validation metrics to dictionary
        """
        # Transfer data to cpu - loss only
        _loss =[] 
        for _item in training_dict['loss']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _loss.append(_item)
        training_dict['loss'] = _loss              
        _loss =[] 
        for _item in validation_dict['loss']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _loss.append(_item)
        validation_dict['loss'] = _loss   

        # Generate writer for a given model      
        _writer = pd.ExcelWriter(self.results_output_dir+ f"{self.model_params.model_name}_{self.model_params.opt_name}_" +  
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}" + 
                    f"_{best_epoch}" + f"{best_score:5f}.xlsx", engine = 'xlsxwriter')

        # Generate dataframes
        _df_train = pd.DataFrame.from_dict(training_dict)
        _df_valid = pd.DataFrame.from_dict(validation_dict)

        _df_train.to_excel(_writer, sheet_name="Training", index = False)
        _df_valid.to_excel(_writer, sheet_name="Validation", index = False)
        _writer.close() 

    #*******************************************************#
    # Main function for training
    #*******************************************************#
    def start_training(self):
        """
        Function which controls training of the model
        """
        print(f"TRAINING STARTED: Epochs: {self.model_params.epochs}, Validation after: {self.model_params.valid_epochs}")

        # Set dictionaries for metrics
        _training_results_dict = {
            'epoch' : [], 'loss' : [], 
            'precision' : [], 'recall' : [], 
            'f1-score' : [], 'precision-w' : [], 
            'recall-w' : [],  'f1-score-w' : [],
        }

        _valid_results_dict = {
            'epoch' : [], 'loss' : [], 
            'precision' : [], 'recall' : [], 
            'f1-score' : [], 'precision-w' : [], 
            'recall-w' : [],  'f1-score-w' : [],
        }
        
        # Set score 
        _best_loss = 1000.0
        _best_epoch = 0

        # Validation params
        _mod_valid = int(self.model_params.valid_epochs.split('_')[0])
        _threshold_valid = int(self.model_params.valid_epochs.split('_')[1])

        # Set scheduler
        if self.model_params.scheduler == 'ReduceLROnPlateau':
            _scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',  factor = 0.1, patience = 3,)	

        # Start training
        for _epoch in range(1, self.model_params.epochs +1):
            print("--------------------------------------")
            print(f"Epoch {_epoch} / {self.model_params.epochs}")
            
            # Run training function
            _start = time.time()
            _predictions, _true, _loss = self.train_one_epoch(self.train_dl)
            _end = time.time()
            
            # Run evaluation function
            _epoch_loss = self.eval_metrics(epoch = _epoch, 
                                            predictions = _predictions, 
                                            true = _true, 
                                            loss = _loss, 
                                            mode = 'train', 
                                            save_dict = _training_results_dict)
            
            # Report to Wandb
            if self.model_params.wandb == True:
                _wandb_metrices = {'Train-loss': _epoch_loss,
                                'Train-precission': _training_results_dict['precision'][-1],
                                'Train-recall': _training_results_dict['recall'][-1],
                                'Train-f1-score': _training_results_dict['f1-score'][-1],
                                'Train-precission-weighted': _training_results_dict['precision-w'][-1],
                                'Train-recall-weighted': _training_results_dict['recall-w'][-1],
                                'Train-f1-score-weighted': _training_results_dict['f1-score-w'][-1]
                }
            
            # Report time
            print(f"Time: {(_end-_start):5f}sec")
            # Save model
            if _epoch % self.model_params.save_epochs == 0:
                self.save_model(_epoch, best = False, info = 'train')
            
            # Validation
            print("######################################")
            if _epoch == 1 or _epoch % _mod_valid == 0 or _epoch >= _threshold_valid:
                _predictions_v, _true_v, _loss_v = self.validate_model(self.valid_dl)
                _epoch_loss = self.eval_metrics(_epoch, _predictions_v, _true_v, 
                                            _loss_v, 'valid', _valid_results_dict)
                # Save best model
                if _epoch_loss < _best_loss:
                    self.save_model(_epoch, best = True, info = 'valid')
                    _best_loss = _epoch_loss
                    _best_epoch = _epoch

                # Report to wandb
                if self.model_params.wandb == True:
                    _validation_metrices = {'Valid-loss': _epoch_loss,
                                'Valid-precission': _valid_results_dict['precision'][-1],
                                'Valid-recall': _valid_results_dict['recall'][-1],
                                'Valid-f1-score': _valid_results_dict['f1-score'][-1],
                                'Valid-precission-weighted': _valid_results_dict['precision-w'][-1],
                                'Valid-recall-weighted': _valid_results_dict['recall-w'][-1],
                                'Valid-f1-score-weighted': _valid_results_dict['f1-score-w'][-1]
                    }
                    _wandb_metrices.update(_validation_metrices)
            
            # Step the scheduler after each epoch
            _scheduler.step(_epoch_loss)
                
            # Wandb report
            if self.model_params.wandb == True:
                wandb.log(_wandb_metrices)

            # Cosmetic
            print("######################################")

            # Early stopping
            if _best_epoch + self.model_params.early_stopping <= _epoch:
                print(f"Early stopping at epoch: {_epoch}")
                break
        
        # Save metrics
        self.export_metrics_to_xlsx(_best_epoch, _best_loss, 
                            _training_results_dict, _valid_results_dict)

        # Release memory
        torch.cuda.empty_cache() 

        # Finish wandb
        if self.model_params.wandb == True:
            wandb.finish()
            
            
    #*******************************************************#
    # Methods used for models predictions
    #*******************************************************#
    def model_predict_from_dl(self, input_data_loader, save_name:str):
        """
        Function is predicting results for the given dataloader and stores them
        
        Input args:
            * input_data_loader, pytorch dataloader, dataloader for which the results are going to be predicted
            * save_name, str, name of the file where the results are going to be stored
        """

        # Notice
        print("######################################")
        print(f"Predicting results on the dataloader")

        # Storage
        _results_dict = {
            'epoch' : [], 'loss' : [], 
            'precision' : [], 'recall' : [], 
            'f1-score' : [], 'precision-w' : [], 
            'recall-w' : [],  'f1-score-w' : [],
        }

        # Obtain results
        _predictions, _true, _loss = self.validate_model(input_data_loader)
        
        # Evaluate metrics
         # Transfer to cpu
        _predictions = _predictions.to('cpu')
        _true = _true.to('cpu')

        # Get 1D array
        _predictions = torch.flatten(_predictions)
        _predictions = torch.where(_predictions > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        _true = torch.flatten(_true)

        # Calculate precission recall f1 score
        _report = metrics.classification_report(_predictions, _true, digits=3, zero_division = 0, output_dict = True)

         # Generate writer for a given model      
        _writer = pd.ExcelWriter(self.results_output_dir+ f"{self.model_params.model_name}_{self.model_params.opt_name}_" +  
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}_" + 
                    save_name+ ".xlsx", engine = 'xlsxwriter')

        # Generate dataframes
        _df_report = pd.DataFrame(_report ).transpose()
        _df_predictions = pd.DataFrame()
        _df_predictions['True'] = _true
        _df_predictions['Predicted'] = _predictions

        # Export
        _df_report.to_excel(_writer, sheet_name="Report", index = False)
        _df_predictions.to_excel(_writer, sheet_name="Predictions", index = False)
        _writer.close() 

    
    def model_predict_from_image_full_body(self,
                                  input_image_path:str = None, 
                                  input_xlsx_path = None,
                                  input_cheatsheet_path = None,
                                  image_dimension: int = 224,
                                  label_dimension: int = 35,
                                  label_type: str = "cluster_remaped",
                                  threshold = 0.5)->dict:
        """
        Function which predicts results based on the image. Also if input xlsx is given, then 
        additional results/statistics is returnted. IT IS UGLY!!!!! My eyes, oh my eyes...

        Args:
            * input_image_path, str, mandatory path to the input image
            * input_xlsx_path, str, non - mandatory, will provide additional info such
            * input_cheatsheet_path, str, path to cheat sheet.
            as label volume based on the xlsx data and statistics.
            * image_dimension, int , dimension of the image to be resized.
            * label_dimension, int, dimension of the label usulally defind by data on
            which the model was trained.
            * label_type, str, type of the label based on which the data will be retreived. 
            * threshold, int, threshold which pushes the label to 1 and 0. Can be set to None,
            then raw value is returned.

        Return:
            * output_dict, dict with following information: case_id, image to plot, predicted, true and 
            all label data.
        """
        # Check if the input image path is provided
        assert input_image_path != None, f"Image path must not be empty"
        
        # Obtain case from image path
        _case_name = os.path.basename(input_image_path).split("_exp")[0]

        # Obtain image and zero pad it and evrything necessary as the
        # image was generated by dataloader
        _sample_dict = {"case_name": _case_name,
                "image_path": input_image_path,
                "labels": None
        }
        _dataPreprocessed = ProcessData(sample = _sample_dict, 
                image_dimension = image_dimension,
                label_dimension = label_dimension,
                label_type = label_type
                )
        _image, _, _ = _dataPreprocessed.get_sample()
        
        if input_xlsx_path != None:
           # Extract labels
            _data_df = pd.read_excel(input_xlsx_path)
            # Obtain blacklist
            _cheatsheet_df = pd.read_excel(input_cheatsheet_path)
            _wrong_labels = np.array(_cheatsheet_df["Label"].tolist()) * np.array(_cheatsheet_df["Keep"].tolist())
            _wrong_labels = np.where(_wrong_labels == 0)[0]
            _wrong_labels += 1
            _blacklist = list(_wrong_labels)

            # Obtain data by _reducted_image sufix TODO
            _query_name = os.path.basename(input_image_path).split("_reducted_image")[0]
            _row = _data_df.loc[_data_df['ID'] == _query_name]

            # Get name dictionary of cheatsheet
            _name_dict = _cheatsheet_df.set_index('Label')['Name'].to_dict()
            
            # Obtain labels and areas
            _labels = []
            for _id in range(1,len(_name_dict.keys())):
                # APPLY BLACKLIST
                if _id in _blacklist:
                    continue
                _area = _row[_id].values[0]
                # Check if label exists _area not nan 
                if pd.isna(_area):
                    continue
                
                # label name
                _original_label_name = _id

                # string label name
                _string_name = _name_dict[_id]

                # Derived names
                _data_cnf = datasetConfig()
                _remap_dict = _data_cnf.remap_dict 
                _prefix_gathered_name, _remaped_name = _remap_dict[_id]

                # Add label to list
                _labels.append({
                    "original_label": _original_label_name,
                    "label_name": _string_name,
                    "cluster_remaped": _prefix_gathered_name,
                    "remaped": _remaped_name,
                    "area": _area
                })
            
            # Obtain tensor labels
            _label_list = []
            # Get indices
            for _label in _labels:
                _sub_label = _label[label_type]
                _label_list.append(int(_sub_label))
            
            # Obtain area
            _areas = []
            for _label in _labels:
                _areas.append(_label["area"])
            # Remove doubles
            # In areas by calculating their mean
            if label_type == "cluster_remaped":
                # Obtain indexes of the same labels
                _index_dict = {}
                for _index, _value in enumerate(_label_list):
                    if _value not in _index_dict:
                        _index_dict[_value] = []
                    _index_dict[_value].append(_index)
                _new_areas = []
                for _key in _index_dict:
                    _sum = 0
                    for _item in _index_dict[_key]:
                        _sum += _areas[_item]
                    _new_areas.append(_sum/len(_index_dict[_key]))
                _areas = _new_areas
            # In label list so that labels are signualar
            _label_list = list(set(_label_list))

            # Reduce by 1 to obtain indexes
            _np_label_list = np.array(_label_list) - 1

            # Output np_array
            _output_np_array = np.zeros(label_dimension)
            _output_np_array[_np_label_list] = 1
            _output_np_array = _output_np_array.flatten()
        
        # Predict
        _image = torch.from_numpy(_image)
        _image = _image.to(device = self.device)
        _image = _image.float()
        _image /= 255.0
        _image = self.aug_model_color_only(_image.unsqueeze(0))

        self.model.eval()
        _prediction = self.model(_image).cpu()

        if threshold != None:
            _prediction = torch.where(_prediction > threshold, torch.tensor(1.0), torch.tensor(0.0))

        _prediction = _prediction.cpu().squeeze(0).numpy()
        _prediction = _prediction.flatten()

        # Return data
        if input_xlsx_path != None:
            return {
                "image_path": input_image_path,
                "id": _case_name,
                "prediction":  _prediction,
                "true": _output_np_array,
                "original_labels": _labels,
                "area": _areas,
                "image": _image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            }
        else:
            return {
               "image_path": input_image_path,
               "id": _case_name,
               "prediction":  _prediction, 
               "image": _image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            }

    def model_predict_from_dl_plain_prediction(self, input_data_loader)->dict:
        """
        Method which returns list of predictions, true labels and paths for statistic caclucatin purposes.

        Args: 
            * inpyt_data_loader, instance of torch data loader, dataloader
        
        Returns:
            * dictionary with True, Predicted and Paths list
        """
        # Create storage dict
        _storage_dict = {"True": [],
                        "Pred": [],
                        "Paths": [],
                        "Datasets": []}

        # Set model to eval mode
        self.model.eval()
        for _batch in input_data_loader:
             # Parse _batch
            _image, _label, _path, _dataset = _batch
            # Transfer data to GPU
            _input_data = _image.to(self.device, non_blocking = True)
            
            # If augumentation is required only for color scheme (validation set)
            if self.aug_model != None:
                _input_data = self.aug_model_color_only(_input_data)

            # Obtain prediction
            with torch.no_grad():
                _predictions = self.model(_input_data)	

            # Convert to _numpy
            _predictions_numpy = _predictions.cpu().numpy()
            _label_numpy = _label.cpu().numpy()

            # Convert numpy to list
            _predictions_list = _predictions_numpy.tolist()
            _label_list = _label_numpy.tolist()

            # Store the finidngs
            _storage_dict["True"] += _label_list
            _storage_dict["Pred"] += _predictions_list
            _storage_dict["Paths"] += _path
            _storage_dict['Datasets'] += _dataset
    
        # Return dict
        return _storage_dict