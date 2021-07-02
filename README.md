
# Visualizing the decision process of GNN computer-aided Alzheimer's Disease diagnosis

The aim of this work is to remove the information barrier created by cutting edge systems computer-aided diagnostic (CAD)
systems within a clinical environment — providing the results of models in a clear and interpretable format for clinicians
and their patients.

The focus of this study begins by using a supervised Graph-Neural Network to predict current condition of a patient's Alzheimer's disease diagnosis using data and labels provided by the [*Alzheimer's Disease Neuroimaging Initiative*](http://adni.loni.usc.edu/).

Neuropsychological assessment, demographic and medical datasets were combined alongside PET analysis of the *ADNI-3* dataset using the *PETPVE12* toolbox provided by Dr. Sanchez-Bornot, *University of Ulster*. Thereafter, the results are interpreted and formatted using a variety of visualization techniques, such as:

- Graphs
- Histograms
- 2-D and 3-D scatter plots

Research on computer-aided diagnosis has been covered extensively, however, unsupervised learning and the interpretation of contrbituting factors remains limited. Using UMAP clustering, datapoints were projected onto 3-Dimensional space and placed inside boundaries where clear delineations of the data occurred. Labels were re-assigned to the resulting datapoints using these boundaries.

![newplot (4)](https://user-images.githubusercontent.com/45361366/124305692-55a12e00-db5d-11eb-9351-f73f12f8f9ee.png)

By segragating the dimensionally reduced data, ranges were provided and illustrated in an attempt to highlight the possible (and notable) factors that may contribute to AD diagnosis.


## Acknowledgements

This code is developed on the code base of:
- [AutoMetric GNN](https://github.com/SJTUBME-QianLab/AutoMetricGNN)
- [Few-shot GNN](https://github.com/vgsatorras/few-shot-gnn)
- [Population GNN](https://github.com/parisots/population-gcn)

Without the great work of these authors, none of this would have been possible given the time constraints — thank you.
