# Clustering of distribution feeders with Interactive Visualizations

- k-means clustering algorithm implementation: to identify 7-10 representative feeders from 700+ distribution feeders
- The algorithm tests k=7, k=8, k=9 etc.. and depending on the Silhouette values for given k, the best number of representative feeders is chosen.
- The code includes the Principal Component Analysis (PCA) of the dataset as well as 3-D visualizations of the data. 
Towards the end, scatter matrix plots are also demonstrated. 

## How to Run
  ### Prerequisites 
  You will need a python 3 interpreter with the following packages:
  - jupyter notebook  
  - scikit learn
  - scipy
  - matplotlib
  - pandas
  (optional - if you want to run 3-D visualization and interactive plot part)
  - plotly
  - cufflinks
  
  ### Running the code
  1. Clone (or download) the repository: 
  
  `git clone https://github.com/sakshi-mishra/clustering-distribution-feeders.git`
  
  2. Open/Run the jupyter notebook [Clustering_of _distribution_feeders_with_Interactive_Visualizations.ipynb](Clustering_of_distribution_feeders_with_Interactive_Visualizations.ipynb)
  

