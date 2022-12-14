hls4ml Live Demo:

Anomaly Detection in Machine Operating Sounds (ADMOS) based on hls4ml ad08 model in mlperf tiny v0.7 submission


steps:

training:

1. run conda env create -f environment.yml and conda activate tiny
2. run get_data.sh to download ToyADMOS dataset and get processed_data
3. python train.py -c ad08.yml to train nn
4. python test.py -c ad08.yml to test auc

inference:

5. python deployment.py -c ad08_pynq.yml to sythesize and profile hls4ml nn

note: 

if inference fails, possible that LUT utilization is too high. try rerunning the steps above 

you can check the LUT util under vivado/myproject_prj/solution1/syn/report/myproject_csynth.rpt

vivado:

6. git clone https://github.com/Xilinx/PYNQ
7. mv audio_nn PYNQ/boards/Pynq-Z2
8. cd PYNQ/boards/Pynq-Z2/audio_nn
9. vivado -mode batch -source project.tcl to sythesize and build bitfile with audio module (make sure vivado file is generated from deployment.py)

pynq_live_admos:

10. move live_inference.ipynb, axi_stream_driver.py, pynq_common.py, hwh and bit file to pynq z2
11. run live_inference.ipynb on pynq z2

note: to set up pynq-z2 please refer to https://pynq.readthedocs.io/en/latest/getting_started/pynq_z2_setup.html

huge thanks to: 
sioni summers (thesps)
jules muhizi (julesmuhizi)
