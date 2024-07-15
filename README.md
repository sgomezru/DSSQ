### Domain Shift Segmentation Quality
<b>Actual title:</b> Estimating image segmentation quality under domain shift via dimensionality reduction in feature space (in the context of MRI)

### Tasks
#### Configuration
1. [x] Define datasets and code their dataloaders
2. [x] Define model and code their initialization
3. [x] Set up and code training loop
4. [x] Set up logging system
#### Phase 1 - Add-on PCA module replication and comparison
1. [x] Train base model
2. [x] Code flexible module attachment in base model
3. [x] Include PCA module in bottleneck
4. [x] Train PCA attached model (Probably requires intermediate step of Mahalanobis distance evaluation)
5. [] Compare results especially regarding OOD data
#### Phase 2 - Other variants
1. [] Train and test out with other dimensionality reduction modules (tSNE, LDA, Avg-Pool)
2. [] Test other anomaly detection methods
3. [] Apply known distortions to inputs
4. [] Autoencoders replica comparison
5. [x] Test out dim reduction in different and/or multiple possible layers
6. [] Analyze and conclude on obtained results

### Bibliography
Look at bib file
