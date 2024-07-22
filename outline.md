## Thesis manuscript outline
#### Estimating image segmentation quality under domain shift via dimensionality reduction in feature space
1. Abstract
2. Introduction
- Mention the problem: Domain shift (In particular to our context? - i.e. MRI from different scanners)
- Previous work here? Mention the paper
3. Methods
- Datasets: Multisite prostate MRI & Heart MNMS
- Upstream task: Segmentation
    - Model (UNet, SWINUNet)
- Downstream task: OOD detection
    - Dim reduction in feature space based on paper
    - Dimensionality reduction & Mahalanobis distance
        - PCA, Incremental PCA, Avg pooling
- Evaluation process
    - Experiments with different number of components
    - Setting of the OOD and ID data
- Attaching dimensionality reduction models in multiple layers
- Mention any implementation details?
4. Results
- Base model for upstream task segmentation results (Both ID and OOD examples)
- Dice score and mahalanobis distace (ood detection) correlation (For the different dimensionality reduction methods)
    - "Plots" (E.g. box plot) that show ID and OOD distributions
    - Correlation scores & error rejection curves (AURCC - Area under risk coverage)
- Additional somewhere: Codebase
5. Discussion
- Contrast results of multiple dimensionality reduction methods wrt. ood detection
- Are results related/expected as per base paper?
6. Conclusion
7. Appendix
- Instructions to replicate? Or is README in Github repo sufficient? Could I just mention that instructions are there and point towards it?
- Extra results figures if there happen to be too many
