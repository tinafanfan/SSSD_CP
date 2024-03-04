# SSSD


## Dataset
1. The NYISO dataset can be downloaded from https://www.nyiso.com/.
2. The cleaned data created by the author can be downloaded from [link](https://drive.google.com/drive/folders/1dwPkBIHSikhQ5ru3HPQiILSnaGAtP3Yr?usp=sharing).


Note that the cleaned data is created following the scripts in dataset_script/nyiso-csv-to-pickle.ipynb and dataset_script/nyiso-load-pickle-to-npy.ipynb.


## Implement
1. Train the model: `python3 train.py -c config/config_SSSDS4-NYISO-3-mix.json`
2. Generate one prediction for each sample in test data: `python3 inference.py -c config/config_SSSDS4-NYISO-3-mix.json --num_samples=803`
3. Generate 10 predictions for each sample in test data: `python3 inference_multiples.py -c config/config_SSSDS4-NYISO-3-mix.json`


## Seggestion
1. Use `CUDA_VISIBLE_DEVICES` to specify the number of GPUs. Both training and inference require the same number of GPUs.
2. Use the sample size as the parameter `--num_samples` in the inference section.


