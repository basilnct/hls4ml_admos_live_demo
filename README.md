hls4ml Live Demo:

Anomaly Detection in Machine Operating Sounds (ADMOS)


steps:

training:

1. run conda env create -f environment.yml and conda activate tiny
2. run get_data.sh to download ToyADMOS dataset and get processed_data
3. run train.py to train nn
4. run test.py to test auc

inference:

5. run deployment to sythesize and profile hls4ml nn

vivado:

6. run design.tcl to sythesize and build bitfile with audio module (WIP)

pynq_live_admos:

7. move live_inference.ipynb, axi_stream_driver.py, pynq_common.py, hwh and bit file to pynq z2
8. run live_inference.ipynb
