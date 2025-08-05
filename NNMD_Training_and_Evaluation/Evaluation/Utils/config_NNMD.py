#*******************************************************#
# Classes used as configuration files
#*******************************************************#  

class datasetConfig:
    '''
    Class for defining dataset parameters: shape, split ratio, augumentation etc.
    '''
    '''
    labes_xlsx_path: Path to xlsx file where export of totalSegmentor is storage
    '''
    labels_xlsx_path = r"C:\Users\CH258598\Desktop\Current Research\BodyPartTraining\Dataset\Output-Labels.xlsx"

    '''
    cheetsheet_xlsx_path: Path to xlsx file which contains instructions on which data to keep
    '''
    cheetsheet_xlsx_path = r"C:\Users\CH258598\Desktop\Current Research\BodyPartTraining\Dataset\cheetsheet.xlsx"

    '''
    Imgs_png_home_path: Root directory where the images are located
    '''
    imgs_png_home_path = r"C:\Users\CH258598\Desktop\Current Research\BodyPartTraining\Dataset\Images"

    '''
    blacklist: Crucial for bone only, if set to False please check remapdict because it is 
    strongly tied to the blacklisting. Function to check: obtain_structured_data in dataloader_NNMD
    '''
    blacklist = True

    '''
    image_sufix: set images sufix so it makes it easier to find them in the image dir. 
    Default sufix is _reducted_image
    '''
    image_sufix = "_reducted_image"

    '''
    type: type of the dataset being builded: train, test, valid
    '''
    type = "train"
    
    '''
    label_type: label type style: it can be cluster_remaped or remaped
    '''
    label_type = "cluster_remaped"

    '''
    partition: float, partition * len(dataset). Defines what partition of the given
    dataset will be used # TODO
    '''
    partition = None
    
    '''
    remap_dict: dictionary for rempaing the labels. This was done based on image provided at (from skull to foot for bones)
    https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/overview_classes_v2.png
    '''
    remap_dict =  {
        91: [1, 1, 1],
        73: [2.1, 2, 2],
        74: [2.2, 3, 2],
        71: [3.1, 4, 2],
        72: [3.2, 5, 2],
        69: [4.1, 6, 3],
        70: [4.2, 7, 3],
        50: [5.1, 8, 4],
        49: [5.2, 9, 4],
        48: [5.3, 10, 4],
        47: [5.4, 11, 4],
        46: [5.5, 12, 4],
        45: [5.6, 13, 4],
        44: [5.7, 14, 4],
        116:[6, 15, 5],
        92: [7.1, 16, 5],
        104: [7.2, 17, 5],
        93: [8.1, 18, 5],
        105: [8.2, 19, 5],
        94: [9.1, 20, 5],
        106: [9.2, 21, 5],
        95: [10.1, 22, 5],
        107: [10.2, 23, 5],
        96: [11.1, 24, 5],
        108: [11.2, 25, 5],
        97: [12.1, 26, 5],
        109: [12.2, 27, 5],
        98: [13.1, 28, 5],
        110: [13.2, 29, 5],
        99: [14.1, 30, 5],
        111: [14.2, 31, 5],
        100: [15.1, 32, 5],
        112: [15.2, 33, 5],
        101: [16.1, 34, 5],
        113: [16.2, 35, 5],
        102: [17.1, 36, 5],
        114: [17.2, 37, 5],
        103: [18.1, 38, 5],
        115: [18.2, 39, 5],
        43: [19.1, 40, 5],
        42: [19.2, 41, 5],
        41: [19.3, 42, 5],
        40: [19.4, 43, 5],
        39: [19.5, 44, 5],
        38: [19.6, 45, 5],
        37: [19.7, 46, 5],
        36: [19.8, 47, 5],
        35: [19.9, 48, 5],
        34: [19.11, 49, 5],
        33: [19.12, 50, 5],
        32: [19.13, 51, 5],
        31: [20.1, 52, 6],
        30: [20.2, 53, 6],
        29: [20.3, 54, 6],
        28: [20.4, 55, 6],
        27: [20.5, 56, 6],
        26: [21, 57, 8],
        124: [22, 58, 7],
        125: [23, 59, 7],
        77: [24.1, 60, 8],
        78: [24.2, 61, 8],
        25: [25, 62, 8],
        75: [26.1, 63, 9],
        76: [26.2, 64, 9],
        126: [27, 65, 10],
        127: [28, 66, 10],
        128: [29, 67, 10],
        118: [30, 68, 11],
        119: [31, 69, 12],
        120: [32, 70, 12],
        121: [33, 71, 13],
        122: [34, 72, 14],
        123: [35, 73, 14],
    }

    body_remap_dict = {
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
    image_dimension = 240

    '''
    label_dimension: label dimension is a length of the label which can be read out
    from remapdict
    '''
    label_dimension = 70


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
        _retval += (f'labels_csv_path: {datasetConfig.labels_xlsx_path}\n'+ 
            f'imgs_png_home_path: {datasetConfig.imgs_png_home_path}\n'+
            f'cheatsheet_xlsx_path: {datasetConfig.cheetsheet_xlsx_path}\n' + 
            f'blacklist: {datasetConfig.blacklist}\n' + 
            f'image_sufix: {datasetConfig.image_sufix}\n' +
            f'type: {datasetConfig.type}\n' +
            f'label_type: {datasetConfig.label_type}\n' +
            f'image_dimension: {datasetConfig.image_dimension}\n' + 
            f'label_dimension: {datasetConfig.label_dimension}\n' +
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
    model_name = 'vgg'

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
              f'wandb: {modelConfig.wandb}\n')
        _retval += ("######################################\n")
        return _retval
