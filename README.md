### Domain Shift Segmentation Quality
<b>Actual title:</b> Estimating image segmentation quality under domain shift via dimensionality reduction in feature space (in the context of MRI)

#### STILL TO BE UPDATED THIS README FOR INSTRUCTIONS OF EASY LAUNCH OFF

#### 20.09.2024

Being inside the cloned repository directory, switch directories to the "docker" directory, and run the following command in the terminal to build the docker image:
```bash
sh build.sh -t dssq
```

I will still update this README for giving instructions on the how to replicate, for now:

- To train the models: src/training_models.py

- To train the adapters after having the models: src/training_adapters.py

- To collect the data (perform all experiments): src/get_data.py or src/get_data_scanner.py

#### Weights and experiment collected data

[Drive with model weights and collected data](https://drive.google.com/drive/folders/1zDsbSquuFGKFuvTujU_OwrbsECdbWGHz?usp=sharing)

#### Datasets

- [MSPMRI](https://liuquande.github.io/SAML/)
- [M\&Ms-2](https://www.ub.edu/mnms-2/)
