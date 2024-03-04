# SSSD

NYISO cleaned data: [dounload](https://drive.google.com/drive/folders/1dwPkBIHSikhQ5ru3HPQiILSnaGAtP3Yr?usp=sharing)

Train the model:
`python3 train.py -c config/config_SSSDS4-NYISO-3-mix.json`

Generate one prediction for each sample in test data:
`python3 inference.py -c config/config_SSSDS4-NYISO-3-mix.json`

Generate 10 predictions for each sample in test data:
`python3 inference_multiples.py -c config/config_SSSDS4-NYISO-3-mix.json`

