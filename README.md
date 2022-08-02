hls4ml Live Demo:

Anomaly Detection in Machine Operating Sounds (ADMOS)


steps:

training:

1. run conda env create -f environment.yml and conda activate tiny
2. run get_data.sh to download ToyADMOS dataset and get processed_data
3. run train.py with ad08.yml to train nn
4. run test.py  with ad08.yml to test auc

inference:

5. run deployment.py with ad08_pynq.yml to sythesize and profile hls4ml nn (not building)

vivado:

6. git clone https://github.com/Xilinx/PYNQ and move audio_nn into PYNQ/boards/Pynq-Z2
7. run design.tcl to sythesize and build bitfile with audio module (it will auto import ip from vivado)

pynq_live_admos:

8. move live_inference.ipynb, axi_stream_driver.py, pynq_common.py, hwh and bit file to pynq z2
9. run live_inference.ipynb
