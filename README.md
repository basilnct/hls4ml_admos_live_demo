Live demo of hls4ml:

Anomaly Detection in Machine Operating Sounds (ADMOS)

steps:
1. run get_data.sh to download ToyADMOS dataset and get processed_data
2. run train.py to train nn
3. run test.py to test auc
4. run deployment to sythesize and profile hls4ml nn
5. run design.tcl to sythesize and build bitfile with audio module
6. move live_inference.ipynb, axi_stream_driver.py, pynq_common.py, hwh and bit file to pynq z2
7. run live_inference.ipynb
