# clustering-distribution-feeders

I have implemented k-means clustering algorithm to identify 7-10 representative feeders from 700+ distribution feeders. The overall objective of the project is to model the PV hosting capacity of the feeders of a given distribution utility system. Given the time and resource constraints, not all the feeders will be modeled in detail to determine the hosting capacity. Thus, the clustering algorithm is implemented to first group (cluster) all the feeders and then to pick the representative ones for detailed modeling.

The algorithm tests k=7, k=8, k=9 etc.. and depending on the Silhouette values for given k, the best number of representative feeders is chosen.

The code includes the Principal Component Analysis (PCA) of the dataset as well as 3-D visualizations of the data. 
Towards the end, scatter matrix plots are also demonstrated. 
