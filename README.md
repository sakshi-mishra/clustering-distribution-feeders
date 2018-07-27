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
  
  3. Open/Run the .py file [Clustering_with_UserDefinedOptions_csvResults.py](Clustering_with_UserDefinedOptions_csvResults.py)
 
###  User Instructions (for running Clustering_with_UserDefinedOptions_csvResults.py file):
####  NOTE: Check that raw values make it into the .csv (not standardardized/normalized values)

### Input to the program:
    1. names of the csv files (more than 1 file/dataset) can be processed at a time.
        Note: files names shouldn't have spaces in between them, you may fill the spaces with underscores.
    2. user-defined transformation method: choose from normalization, standardization and minmax-scaling (based on dataset)
    3. number of expected cluster: a range of k values (number of clusters to form) can be given as input. If just one value of k to be tested, then enter a = b
    
### Output: 
    1. csv file named "input_file_name+transformation+date+data" : contains the labels assinged to all the samples
    2. csv file named "input_file_name+transformation+date+metric" : contains various evaluation metircs
    3. individual scatter plots of the features (for each cluster)
    4. silhouette value plot
