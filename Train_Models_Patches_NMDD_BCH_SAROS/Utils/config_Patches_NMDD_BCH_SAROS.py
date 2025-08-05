#*******************************************************#
# Classes used as configuration files
#*******************************************************#  

class datasetConfig:
    '''
    Class for defining dataset parameters: shape, split ratio, augumentation etc.
    '''
    '''
    mapping_xlsx_path_NMDD: Path to directories where each dir has results and original image in it
    '''
    mapping_xlsx_path_NMDD = r"C:\Users\CH258598\Desktop\Current Research\BodyPartTraining\Dataset\Output-Labels.xlsx"

    '''
    batch_xlsx_path_NMDD: Path to the parent dir where "batch_x.xlsx" files are stored
    '''
    batch_xlsx_path_NMDD = None

    '''
    mapping_xlsx_path_SAROS: Path to directories where each dir has results and original image in it
    '''
    mapping_xlsx_path_SAROS = r"C:\Users\CH258598\Desktop\Current Research\BodyPartTraining\Dataset\Output-Labels.xlsx"

    '''
    batch_xlsx_path_SAROS: Path to the parent dir where "batch_x.xlsx" files are stored
    '''
    batch_xlsx_path_SAROS = None

    
    '''
    dir_path_BCH: Path to the directory holding exported labels
    '''
    dir_path_BCH = r"C:\Users\CH258598\Desktop\FiteredDatasetNNMD256Subset\DataGenerationScripts\BCH_VERIFED"

    '''
    dir_path_NMDD: path to the 275 verified NMDD data instances
    '''
    dir_path_NMDD = r"C:\Users\CH258598\Desktop\FiteredDatasetNNMD256Subset\CheckedOutNNMD"

    '''
    dir_path_SAROS: path to the 299 verified SAROS data instances
    '''
    dir_path_SAROS = r"C:\Users\CH258598\Desktop\FiteredDatasetNNMD256Subset\CheckedOutNNMD"

    '''
    type: type of the dataset being builded: train, test, valid
    '''
    type = "train"
    
    '''
    partition: float, partition * len(dataset). Defines what partition of the given
    dataset will be used # TODO
    '''
    partition = None
    
    '''
    remap_dict: dictionary for rempaing the labels. This was done based on image provided at (from skull to foot for bones)
    https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/overview_classes_v2.png
    '''
    remap_dict = {
        "skull": ["skull"],
        "shoulder": ["clavicula" ,"scapula"], 
        "humerus": ["humerus"],
        "vertebrae_C": ["vertebrae_C"],
        "thorax": ["vertebrae_T", "sternum", "rib_1", "rib_2", "rib_3", "rib_4", "rib_5", "rib_6", "rib_7", "rib_8", "rib_9", "rib_10", "rib_11", "rib_12"],
        "vertebrae_L": ["vertebrae_L"],
        "forearm": ["ulna", "radius"], 
        "pelvis": ["hip", "vertebrae_S", "sacrum"],
        "femur": ["femur"],
        "hand": ["carpal", "metacarpal", "phalanges_hand"], 
        "patella": ["patella"],
        "shin": ["tibia", "fibula"],
        "tarsal": ["tarsal"],
        "foot":["metatarsal", "phalanges_feet"]
    }
    
    '''
    image_dimension: Image size, Single integer controling the shape of the input
    image. image_dimension = 200 means that input image will be 200x200 pixels 
    '''
    image_dimension = 300

    '''
    split_ratio: split ratio for train test valid. train subset = split ratio, 
    valid and test subsets = (1 - split_ratio)/2
    '''
    split_ratio = 0.8
    
    '''
    folds: list with form of [k, n] where k is k-th fold of n total folds. K is 0 indexed. 
    Default None. If set to a value, ratio is ignored.
    '''
    folds = None

    '''
    Verbose: printing some notifications during debug
    '''
    verbose = True
        
    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'mapping_xlsx_mapping_path_NMDD: {datasetConfig.mapping_xlsx_path_NMDD}\n'+ 
                    f'batch_xlsx_path_NMDD: {datasetConfig.batch_xlsx_path_NMDD}\n'+ 
                    f'batch_xlsx_mapping_path_SAROS: {datasetConfig.mapping_xlsx_path_SAROS}\n'+ 
                    f'batch_xlsx_path_SAROS: {datasetConfig.batch_xlsx_path_SAROS}\n'+ 
                    f'dir_path_BCH: {datasetConfig.dir_path_BCH}\n' +
                    f'dir_path_NMDD: {datasetConfig.dir_path_NMDD}\n' +
                    f'dir_path_SAROS: {datasetConfig.dir_path_SAROS}\n' +
                    f'type: {datasetConfig.type}\n' +
                    f'image_dimension: {datasetConfig.image_dimension}\n' + 
                    f'split_ratio: {datasetConfig.split_ratio}\n' +
                    f'folds: {datasetConfig.folds}\n' +
                    f'verbose: {datasetConfig.verbose}\n')
        
        _retval += ("######################################\n")
        return _retval
   
class loaderConfig:
    '''
    Class for defining loader parameters: batch_size, number of workers
    and gpu
    '''
    '''
    batch_size: define size of the batch
    '''
    batch_size = 32

    '''
    number_of_workers: paralelization for data loading
    '''
    number_of_workers = 4

    '''
    use_gpu: name of the gpu: typicaly 'cuda:0', 'cuda:1' or 'cpu'
    '''
    use_gpu = 'cpu'


    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += "######################################\n"
        _retval += (f'batch_size: {loaderConfig.batch_size}\n'+
              f'number_of_workers: {loaderConfig.number_of_workers}\n' +
              f'use_gpu: {loaderConfig.use_gpu}\n')
        _retval += "######################################\n"
        return _retval
    
class modelConfig:
    '''
    Class for defining model parameters: model name, number of epoch,
    validation ratio, early stopping, optimizer, learning rate, 
    loss, and gpu
    '''
    '''
    name: set model name.
    '''
    model_name = 'eff3'

    '''
    model_input_params: some models requires input data such as input shape. This is a dict
    which can contain all those params.
    '''
    input_shape = None

    '''
    input_channels: number of chanels that we are willing for output image to have.
    Default is 1. Possible value: 0,1,3 (like RGB fashion which will result in copying one channel
    3 times, 0 the channels will not be added)
    '''
    input_channels = 1

    '''
    epochs: number of epochs for training
    '''
    epochs = 150

    '''
    valid_epochs: number which defines when validation will occure. Format is
    "x_y" where model does each "x" epochs validaiton until it reaches "y" epoch. 
    After "y" epoch is reached, validation is done every epoch
    '''
    valid_epochs = '2_10'

    '''
    save_epochs: number after each training model is being saved
    '''
    save_epochs = 5

    '''
    early_stopping: define early stopping besed on the validation epochs
    '''
    early_stopping = 50

    '''
    opt_name: optimizer name. Currently implemented: "ADAMW", "ADAM", "SGD"
    '''
    opt_name = 'ADAMW'

    '''
    learning_rate: set a float which is the inital learning rate.
    Set it to "Auto" to find best lr automatically
    '''
    learning_rate = 10e-3

    '''
    Scheduler: implemented schedulers ReduceLROnPlateau, #CyclicLRWithRestarts
    '''
    scheduler = 'ReduceLROnPlateau'

    '''
    loss_name: name of the loss.
    '''
    loss_name = 'bce'

    '''
    gpu: define which gpu will be use. Chose from names: 'cuda:0', 'cuda:1', 'cpu'
    '''
    gpu = 'cpu'

    '''
    Augumentation model: model given for the augumentation. Several
    example models are located in models.py. E.G.: RGB_transform, TransformXrayGray
    '''
    augmentation_model = None

    '''
    Wandb: True or False if someone wants wandb monitoring
    '''
    wandb = False

    '''
    pretrained: If model wants to be trained from scratch, set this to False, otherwise
    ImageNet weights will be loaded
    '''
    pretrained = False

    '''
    info
    '''
    info = None

    '''
    number of output classes
    '''
    number_of_output_classes = 4

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'name: {modelConfig.model_name}\n'+ 
              f'epochs: {modelConfig.epochs}\n' +
              f'valid_epochs: {modelConfig.valid_epochs}\n'+
              f'early_stopping: {modelConfig.early_stopping}\n'+ 
              f'opt_name: {modelConfig.opt_name}\n'+ 
              f'save_epochs: {modelConfig.save_epochs}\n'+ 
              f'learning_rate: {modelConfig.learning_rate}\n'+ 
              f'loss_name: {modelConfig.loss_name}\n'+ 
              f'augmentation_model: {modelConfig.augmentation_model}\n' +
              f'gpu: {modelConfig.gpu}\n' +
              f'pretrained: {modelConfig.pretrained}\n' +
              f'number_of_output_classes: {modelConfig.number_of_output_classes}\n' +
              f'info: {modelConfig.info}\n' +
              f'input_shape: {modelConfig.input_shape}\n' + 
              f'number_of_output_channels: {modelConfig.input_channels}\n' +
              f'wandb: {modelConfig.wandb}\n')
        _retval += ("######################################\n")
        return _retval
