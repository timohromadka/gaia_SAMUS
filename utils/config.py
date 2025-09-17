# This file is used to configure the training parameters for each task
class Config_US30K:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = "../../dataset/SAMUS/" 
    data_subpath = "../../dataset/SAMUS/ThyroidNodule-TN3K/" 
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-TN3K"  # the file name of training set
    val_split = "val-ThyroidNodule-TN3K"     # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/Breast-BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  # 
    data_subpath = "../../dataset/SAMUS/Echocardiography-CAMUS/" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"



class Config_NICHE:
    """
    Configuration for the custom NICHE dataset.
    """
    data_path = "/net/beegfs/groups/gaia/niche_segmentation_storage/datasets/niches_15_9_sbaf0_pnc100_SAMUS"
    data_subpath = "/net/beegfs/groups/gaia/niche_segmentation_storage/datasets/niches_15_9_sbaf0_pnc100_SAMUS/Niches"
    save_path = "/net/beegfs/groups/gaia/gaia_SAMUS_storage/checkpoints/NICHES/"
    result_path = "/net/beegfs/groups/gaia/gaia_SAMUS_storage/results/NICHES/"
    tensorboard_path = "/net/beegfs/groups/gaia/gaia_SAMUS_storage/checkpoints/NICHES/"
    load_path = save_path + "SAMUS_09171857_8_0.6408096927101944.pth"  # Replace xxx.pth with actual checkpoint filename when resuming
    # load_path = save_path + "sam_vit_b_01ec64.pth"  # Replace xxx.pth with actual checkpoint filename when resuming
    save_path_code = "_"

    # ===== TRAINING PARAMETERS =====
    workers = 2
    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    momentum = 0.9

    # Binary segmentation: background + niche
    classes = 2  

    # Image size for training
    img_size = 256

    # These must match the names of text files in MainPatient/
    train_split = "train"
    val_split = "val"
    test_split = "test"

    # Optional cropping (None = no cropping)
    crop = None

    # Frequency for evaluation and saving
    eval_freq = 1
    save_freq = 2000

    # Hardware
    device = "cuda"
    cuda = "on"

    # Image format
    gray = "yes"        # 'yes' if grayscale images, 'no' if RGB
    img_channel = 1     # 1 for grayscale, 3 for RGB

    # Evaluation mode
    eval_mode = "mask_slice"  # use "camusmulti" if multi-class dataset

    # Whether to use a pretrained SAMUS model for fine-tuning
    pre_trained = True

    # Mode: "train" or "test"
    mode = "train"

    # Visualize outputs during training
    visual = True

    # Model architecture
    modelname = "SAMUS"


# ==================================================================================================
# Factory function to get the configuration for the chosen dataset
def get_config(task="US30K"):
    if task == "US30K":
        return Config_US30K()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "CAMUS":
        return Config_CAMUS()
    elif task == "NICHE":
        return Config_NICHE()
    else:
        raise ValueError("We do not have the related dataset, please choose another task.")
