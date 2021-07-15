# GNN computer-aided Alzheimer's Disease diagnosis


The aim of this work was to breakdown the information barrier created by cutting edge computer-aided diagnostic (CAD)
systems within a clinical environment — providing the results of *Machine Learning* (ML) algorithms in a clear and interpretable format for clinicians
and their patients.

The focus of this study begins with the use of a *Graph-Neural Network* (GNN) to derive predictions from both supervised and unsupervised provisioned training data —
 then used to predict current condition of a patient's Alzheimer's disease (AD) diagnosis based on data and labels provided by the [*Alzheimer's Disease Neuroimaging Initiative*](http://adni.loni.usc.edu/).
 
![Fig4 1](https://user-images.githubusercontent.com/45361366/125794017-a448ab9b-1bf6-4637-a4f1-47309c311adf.png)

Neuropsychological assessment, demographic and medical datasets were combined alongside ROI-based, PET analysis of *ADNI-3* neuroimaging data using the *PETPVE12* toolbox provided by Dr. Sanchez-Bornot, *University of Ulster*. Thereafter, the results were interpreted and formatted using a variety of visualization techniques, such as:

- Graphs
- Histograms
- Clustering methods

Research on computer-aided diagnosis has been covered extensively, however, unsupervised learning and the interpretation of contrbituting AD factors remains limited. Using UMAP clustering, datapoints were dimensionally reduced and projected onto 3-dimensional space in-order to decipher clear delineations between new patterns that occurred. Labels were then re-assigned based on those boundary definitions.

![Fig3](https://user-images.githubusercontent.com/45361366/125796002-cdc5dcbb-b0df-46df-a773-a3955d2fdc4b.png)

By segragating the dimensionally reduced data, ranges were provided and illustrated in an attempt to highlight unconventional trends in recognizing and diagnosing AD. This study could be extended by applying the GNN and data used in this study to a *GNN Explainer*, such as frameworks provided by [Ying *et al.*](https://arxiv.org/abs/1903.03894)


## Acknowledgements

This code has been developed off of the following code bases:
- [AutoMetric GNN](https://github.com/SJTUBME-QianLab/AutoMetricGNN)
- [Few-shot GNN](https://github.com/vgsatorras/few-shot-gnn)
- [Population GNN](https://github.com/parisots/population-gcn)

Without the great work of these authors, none of this would have been possible given the time constraints — thank you.
