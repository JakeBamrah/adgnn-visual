
# GNN computer-aided Alzheimer's Disease diagnosis

The focus of this study begins with the application of heterogeneous *Graph-Neural Networks* (GNN) to derive an Alzheimer's disease (AD) diagnosis using both supervised and unsupervised provisioned data—based on clinical samples provided by the [*Alzheimer's Disease Neuroimaging Initiative*](http://adni.loni.usc.edu/).
 
Neuropsychological assessments, sociodemographic and medical factors were combined alongside ROI-based, PET analysis of *ADNI-3* neuroimaging data. Neuroimages were processed according to a cortical parcellation standard (*Desikan-Killiany template*) using *PETPVE12* toolbox—cortical parcellation was undertaken by [Dr. Sanchez-Bornot](https://www.scopus.com/authid/detail.uri?authorId=9333309700), *University of Ulster*. Thereafter, the preprocessed data were formatted and visualized prior to developing the model architecture.
 
![Fig  4 with coloured background2](https://user-images.githubusercontent.com/45361366/135581812-e5b20424-1807-406c-9e64-64033f4379fd.png)
<sub><sup>Fig. 1. Overview of GNN architecture and training process.</sup></sub>

Research on computer-aided diagnosis has been covered extensively, however, unsupervised learning and the interpretation of contrbituting AD factors remains limited. Using UMAP clustering, each patient feature-set was dimensionally reduced and projected onto 3-dimensional space in an attempt to provide clear delineations between  varying states of the disease. Labels were then re-assigned based on those *new* boundary definitions.

![Fig3-min](https://user-images.githubusercontent.com/45361366/125798044-f48eaa03-00cd-4267-bac5-11a54660760c.png)
<sub><sup>Fig. 2. Dimensionally reduced UMAP clusters with suggested AD bounds.</sup></sub>

The dimensionally-reduced data was segmented and normalized before defining reasonable ranges for each feature, with the aim of exposing novel trends of the disease. Given the limited sample-size, one-shot learning was employed to train the model in batches of 64 samples. Final model performance yielded a mean **89% accuracy** for patient diagnosis using unsupervised cluster-based label assignment—highlighting the potential benefits of unsupervised classification methods in determining AD contributing factors.


## Acknowledgements

This code has been developed off of the following code bases:
- [Few-shot GNN](https://github.com/vgsatorras/few-shot-gnn)
- [Population GNN](https://github.com/parisots/population-gcn)
- [AutoMetric GNN](https://github.com/SJTUBME-QianLab/AutoMetricGNN)

Given the time constraints, none of this would have been possible without the great work of these authors.
