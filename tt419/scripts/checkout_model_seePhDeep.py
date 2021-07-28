# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import torch
import os
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')

from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run
from selene_sdk.utils import DeeperDeepSEA
from selene_sdk.predict._common import predict
from selene_sdk.utils import load_features_list


# %%
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
from scipy.io import loadmat


# %%
sys.path.insert(1,'/home/tt419/Projects/DeepLearning/')
import PhDeep.config.log_conf as log_conf
import PhDeep.models.deepsea_fyr.model as fyr
from PhDeep.data_loader.deepsea.data_loader import DataManager


# %%

batch_size = 64 # Magic number from Selene, used here for consistency
n_valid_samples = 32000 # Magic number from Selene, used here for consistency

# %%
features_504 = load_features_list("/rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Epigenome/output_encode_fold/encode_meta1_features.txt")
features_919 = load_features_list("/rds-d5/user/tt419/hpc-work/data-storage/DeepSea_orig_data/deepsea_train/label_names.txt")
features_1846 = load_features_list("/rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Epigenome/Selene_data/selene_ftp_output/distinct_1846_features.txt")


# %%
# ToDo: This path should be adjusted to make clear, that it is the file for the FYR model/original deepsea (per ln -s)
base_path_919 = "/home/tt419/Projects/DeepLearning/DeepSea_data/deepsea_train/"
train_919 = base_path_919 + "train.mat"
test_919 = base_path_919 + "test.mat"
valid_919 = base_path_919 + "valid.mat"

# ToDo: This path should be adjusted to make clear, that it is the file for the 504 features
base_path_504 = "/rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Epigenome/output_encode_fold/"
data_504 = base_path_504 + "encode_meta1_sorted.bed.gz"
# ToDo: This path should be adjusted to make clear, that it is the file for the 1846 features
base_path_1846 = "/rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Epigenome/Selene_data/selene_ftp_output/"
data_1846 = base_path_1846 + "sorted_selene_fullFeatures.bed.gz"


# %%
model_path_504 = "/rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Epigenome/output_encode_fold/logs/online_sampler_outputs_503_v1_r1/best_model.pth.tar"
#different file-type, as it was run with TF instead of PyTorch
model_path_919 = "/rds-d5/user/tt419/hpc-work/data-storage/PhDeep_logs/DeepSea_ckpt/modelCheckpoint-0.05066-4-99.h5"
model_path_1846 = "/rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Epigenome/Selene_data/selene_ftp_output/logs/online_sampler_outputs_base/best_model.pth.tar"


# %%
ref_hg19 = "/rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Epigenome/Selene_data/selene_ftp_output/male.hg19.fasta"

# %% [markdown]
# # Load model architectures for Selene models

# %%
from selene_sdk.utils import NonStrandSpecific


model_arch_504 = NonStrandSpecific(DeeperDeepSEA(2000, 504))
model_arch_1846 = NonStrandSpecific(DeeperDeepSEA(2000, 1846))


# %%
from selene_sdk.predict import AnalyzeSequences
from selene_sdk.utils import load_features_list

analysis_504 = AnalyzeSequences(
    model_arch_504,
    model_path_504,
    sequence_length=2000,
    features=features_504,
    use_cuda=False)

analysis_1846 = AnalyzeSequences(
    model_arch_1846,
    model_path_1846,
    sequence_length=2000,
    features=features_1846,
    use_cuda=False)


# %% [markdown]
# # Load model for PhDeep

# %%
local = False
load = True
tf.debugging.set_log_device_placement(True)

avail_gpus = tf.config.list_physical_devices('GPU')
no_avail_gpus = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: {} ;\n namely: {}".format(no_avail_gpus, avail_gpus))
for gpu in avail_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

lc = log_conf.LogConfig("/home/tt419/Projects/DeepLearning/PhDeep/logs/", "DeepSea_ckpt/")
model_arch_919 = fyr.DeepSeaModel(sub_version=0.0, log_config=lc)
dm = DataManager()


# %%
if load:
    lc.set_best_checkpoint(model_arch_919.MODEL_VERSION)
    if lc.BEST_CHECKPOINT == -1:
        model_arch_919.build_model()
        load_fp = model_path_919
        print("Load failed, no checkpoint found")
    else:
        load_fp = lc.CHECKPOINT_DIR + lc.BEST_CHECKPOINT
        model_arch_919.load_model(fp=load_fp)
        print(f"Model loaded from {load_fp}")
else:
    load_fp = None

model_919 = model_arch_919.compile_model(multi_gpu=no_avail_gpus>1 ,load_fp=load_fp)
model_919.summary()

# %% [markdown]
# # Load test data
# This data can be used for prediction and put into the "visualize_roc_curves" function of the utils/performance_metrics.py  

# %% [markdown]
# ### Region selection
# It is to be debated, whether the models should all be benchmarked on the same regions, or just on the same chromosomes (usually 6 & 7 or 8 & 9)

# %%
from selene_sdk.sequences import Genome
from selene_sdk.samplers import RandomPositionsSampler
ref_hg19_seq = Genome(ref_hg19)
rps_504 = RandomPositionsSampler(ref_hg19_seq, data_504, features_504, 
                        sequence_length=2000, mode="validate", 
                        save_datasets=["validate"], output_dir=base_path_504+"validation_set_504")

rps_1846 = RandomPositionsSampler(ref_hg19_seq, data_1846, features_1846, 
                        sequence_length=2000, mode="validate", 
                        save_datasets=["validate"], output_dir=base_path_1846+"validation_set_1846")

# dm = DataManager()
# val_x, val_y = dm.read_val_data()
from selene_sdk.samplers.file_samplers import MatFileSampler

mfs_valid_919 = MatFileSampler(valid_919, "validxdata", "validdata")

# %% [markdown]
# ### Sampler ready, now the prediction begins that will be fed into the performance_metrics.py
# The code for the selene-based models is using parts of the "evaluate_model.py" for generating the predictions from the online sampler

# %%
# Predict the validation data with the PhDeep model and pass the predictions to the visualize_roc_curves and visualize_precision_recall_curves

from selene_sdk.samplers import MultiFileSampler
from PhDeep.models.deepsea_selene.model import DeepSEA

weights_919 = model_919.get_weights()
selene_919 = DeepSEA(1000, 919)

layer_map = {
    "conv1" : [model_919.get_layer("conv1d_1"),selene_919.conv_net[0]],
    "conv2" : [model_919.get_layer("conv1d_2"),selene_919.conv_net[4]],
    "conv3" : [model_919.get_layer("conv1d_3"),selene_919.conv_net[8]],
    "dense1" : [model_919.get_layer("dense_1"),selene_919.classifier[0]],
    "dense2" : [model_919.get_layer("dense_2"),selene_919.classifier[2]]  
}

layer_order = ["conv1", "conv2", "conv3", "dense1", "dense2"]

# %%

for layer_key in layer_order:
    layers = layer_map[layer_key]
    layers[1].weight.data=torch.from_numpy(np.transpose(layers[0].weights[0].numpy()))
    layers[1].bias.data=torch.from_numpy(layers[0].weights[1].numpy())

selene_919_model_path = base_path_919+"fyr_to_selene.pth.tar"
import os.path
if not os.path.isfile(selene_919_model_path):
    torch.save(selene_919.state_dict(), selene_919_model_path)
# %%
from torch.nn import BCELoss
from selene_sdk.evaluate_model import EvaluateModel

eval_919 = EvaluateModel(selene_919, BCELoss(), mfs_valid_919, features_919, selene_919_model_path, output_dir=base_path_919+"eval_919", batch_size=batch_size, n_test_samples=n_valid_samples, use_cuda=True)
eval_504 = EvaluateModel(model_arch_504, BCELoss(), rps_504, features_504, model_path_504, output_dir=base_path_504+"eval_504", batch_size=batch_size, n_test_samples=n_valid_samples, use_cuda=True)
eval_1846 = EvaluateModel(model_arch_1846, BCELoss(), rps_1846, features_1846, model_path_1846, output_dir=base_path_1846+"eval_1846", batch_size=batch_size, n_test_samples=n_valid_samples, use_cuda=True)

