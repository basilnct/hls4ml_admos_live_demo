from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

# load model
import os
os.environ['PATH']='/opt/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
model = load_model('model/model_ToyCar.h5', custom_objects=co)

# loading config file
import yaml
convert_config = yaml.safe_load(open('ad08_pynq.yml'))['convert']

# define configuration parameters
BOARD_NAME = convert_config['board_name']
FPGA_PART = convert_config['fpga_part']
TB_DATA_DIR = convert_config['tb_data_dir']
X_TEST_DATA_DIR = convert_config['x_npy_dir']
Y_TEST_DATA_DIR = convert_config['y_npy_dir']
OUTPUT_DIR = convert_config['output_dir']
BACKEND=convert_config['backend']

# convert to hls4ml
import plotting
import hls4ml
hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

# for tuning and custom configs
HLS_CONFIG=convert_config['hls_config']
hls_config = yaml.safe_load(open(HLS_CONFIG))


print("-----------------------------------")
plotting.print_dict(hls_config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=hls_config,
                                                       output_dir=OUTPUT_DIR,
                                                       backend=BACKEND,
                                                       board=BOARD_NAME)
hls_model.compile()
print(hls4ml.templates.get_supported_boards_dict().keys())
plotting.print_dict(hls4ml.templates.get_backend('VivadoAccelerator').create_initial_config())


# profiling / tracing

import numpy as np
import matplotlib.pyplot as plt

X_tb = np.load(TB_DATA_DIR, allow_pickle=True)
X_profiling =  np.load(X_TEST_DATA_DIR, allow_pickle=True)

PROFILE_DIR = OUTPUT_DIR +'/hls4ml_profiling_plots'
os.makedirs(PROFILE_DIR)

hls4ml_pred, hls4ml_trace = hls_model.trace(np.ascontiguousarray(X_profiling[0][0][0]))
# run tracing on a portion of the test set for the Keras model (floating-point precision)
keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_profiling[0][0])

for key in hls4ml_trace:
    plt.figure()
    plt.scatter(keras_trace[key][0], hls4ml_trace[key][0], color='black')
    plt.plot(np.linspace(np.min(keras_trace[key][0]),np.max(keras_trace[key][0]), 10),
            np.linspace(np.min(keras_trace[key][0]),np.max(keras_trace[key][0]), 10), label='keras_range')
    plt.plot(np.linspace(np.min(hls4ml_trace[key][0]),np.max(hls4ml_trace[key][0]), 10),
            np.linspace(np.min(hls4ml_trace[key][0]),np.max(hls4ml_trace[key][0]), 10), label='hls4ml_range')
    plt.title(key)
    plt.xlabel('keras output')
    plt.ylabel('hls4ml output')
    plt.legend()

    plt.savefig(f'{PROFILE_DIR}/{key}')
    print('profiled layer {}'.format(key))

print("-----------------------------------")
print('Plotting AUC Curves')
print("-----------------------------------")

from plot_roc import plot_roc
os.makedirs(f'{OUTPUT_DIR}/test', exist_ok=1)
plot_roc(model, hls_model, X_TEST_DATA_DIR, Y_TEST_DATA_DIR, data_split_factor=1, output_dir=f'{OUTPUT_DIR}/test')


# synthesize
hls_model.build(csim=False, export=True)

# build to check build is successful before adding audio component
hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
