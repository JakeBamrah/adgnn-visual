
# GNN computer-aided Alzheimer's Disease diagnosis

The focus of this study begins with the use of a heterogeneous *Graph-Neural Network* (GNN) to derive Alzheimer's disease (AD) predictions from both supervised and unsupervised provisioned training data —
 being used to predict the current condition of a patient's Alzheimer's disease (AD) diagnosis based on data and labels provided by the [*Alzheimer's Disease Neuroimaging Initiative*](http://adni.loni.usc.edu/) thereafter.
 
Neuropsychological assessments, sociodemographic and medical factors were combined alongside ROI-based, PET analysis of *ADNI-3* neuroimaging data. Neuroimaging data was processed according to cortical parcellation standard (*Desikan-Killiany template*) using *PETPVE12* toolbox — cortical parcellation was undertaken by [Dr. Sanchez-Bornot](https://www.scopus.com/authid/detail.uri?authorId=9333309700), *University of Ulster*. Thereafter, the preprocessed data were formatted and visualized before developing the model architecture.
 
![Fig  4 with coloured background2](https://user-images.githubusercontent.com/45361366/135581812-e5b20424-1807-406c-9e64-64033f4379fd.png)


Research on computer-aided diagnosis has been covered extensively, however, unsupervised learning and the interpretation of contrbituting AD factors remains limited. Using UMAP clustering, datapoints were dimensionally reduced and projected onto 3-dimensional space in-order to decipher clear delineations between new patterns that occurred. Labels were then re-assigned based on those boundary definitions.

![Fig3-min](https://user-images.githubusercontent.com/45361366/125798044-f48eaa03-00cd-4267-bac5-11a54660760c.png)

By segmenting the dimensionally reduced data, ranges were outlined in an attempt to highlight unconventional trends in recognizing and diagnosing AD. Given the limited sample-size, one-shot learning was adopted to train the model in batches of 64 samples. Final model performance yielded a mean 89% accuracy for patient diagnosis using unsupervised cluster-based label assignment — highlighting the potential benefits of utilising unsupervised methods in determining AD contributing factors.


## Acknowledgements

This code has been developed off of the following code bases:
- [AutoMetric GNN](https://github.com/SJTUBME-QianLab/AutoMetricGNN)
- [Few-shot GNN](https://github.com/vgsatorras/few-shot-gnn)
- [Population GNN](https://github.com/parisots/population-gcn)

Without the great work of these authors, none of this would have been possible given the time constraints — thank you.
