

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import pandas as pd
from pandas.tools.plotting import scatter_matrix, parallel_coordinates

print(__doc__)
import datetime
now = datetime.datetime.now()
dtname = now.strftime("%Y%m%d")


# file names
raw_file = input("enter the names of the raw csv files (without the .csv extension) with one space between each name: ")
files = raw_file.split()

# runs =["", "-std", "-minmax", "-norm"] # choose the type of transformation from standardization, normalization, minmax-scaling
runs = input("select and enter the transformation from -std, -minmax, -norm: ")
runs = runs.split()

# choose the number of clusters (can also be multiple clusters i.e. 7,11,1 would give results for 7,8,9,10 clusters)
ran_start = input("enter the value of 'a' for the range(a,b) denoting the range of number of clusters k to be tested: ")
ran_stop = input("enter the value of 'b' for the range(a,b) denoting the range of number of clusters k to be tested: ")
range_k = range(int(ran_start),int(ran_stop)+1,1)

# could instead use np.linalg.norm(x,axis=1)
L2norm = lambda x: np.sqrt(np.sum(np.multiply(x,x),axis=1))
sosq = lambda x: sum(np.multiply(x,x))



for fname in files:
    df = pd.read_csv("D:\\OneDrive\\2017-2018\\Clustering_Analysis\\" + fname + ".csv",sep=",",header=0,low_memory=False)
    df = df.set_index('CIRCUIT')
        
    for rname in runs:
        print('Processing file ' + fname + ', run ' + rname)
        
        X = df.values[:,1:]
        ndim = len(X[0,:])
        nsamp = len(X[:,0])

        # Scale data
        if rname == "-std":
            X = preprocessing.scale(X)
        elif rname == "-minmax":
            X = preprocessing.MinMaxScaler().fit_transform(X)
        elif rname == "-norm":
            X = preprocessing.normalize(X, norm='l2')
        
        all_labels = pd.DataFrame(data = df.index)
        
        
        
        kmetric_labels = {'inertia':'Inertia\n(lower is tighter)', 
            'dunn': 'Dunn score\n(higher is better, prone to noise and outliers)', 
            'db': 'Davies-Bouldin score\n(lower is better, worst-case measure)', 
            'silhouette': 'Average silhouette score\n(higher is more compact and separate)',
            'ch': 'Calinski-Harabasz score\n(higher is better)',
            'ss': 'Sum of Squares\n(lower is better)'}
        kmetric_col = list(kmetric_labels)
        kmetric_col.append('k')
        all_kmetric = pd.DataFrame(data = np.vstack((np.zeros(shape=(len(kmetric_col)-1,len(range_k))),range_k)).T, 
            columns = kmetric_col)
        all_kmetric = all_kmetric.set_index('k')
        
        for k in range_k:
            
            """
            Steps:
            1. Initializing the clusterer with k value (and a random generator)
            2. fitting the model on the dataset (X)
            3. prediction of the labels (i.e. cluster number k) for reach sample/observation
            """
            
            clusterer = KMeans(n_clusters=k, init='k-means++', n_init=100, max_iter=10000, random_state=10)
            KM = clusterer.fit(X)
            cluster_centers = KM.cluster_centers_
            cluster_labels = KM.predict(X)
            all_labels['Cluster for k = ' + str(k)] = cluster_labels
            
            
            """
            1. transform data to a cluster-distance space.
            2. calculation of intracluster and intracluster distance using L2norm distance 
            3. calculation of the distance between the centers of different cluster  
            4. calculation of WCSS (within cluster sum of squares) and of BCSS (between cluster sum of squares)
            5. calcuation of other evaluation metrics: inertia, Dunn, 'Davies-Bouldin score, silhouette,
                Calinski-Harabasz score,  Sum of Squares
                
            """

            # NOTE: KM.transform(X) = cdist(X, cluster_centers, 'euclidean')
            x_dk = X - cluster_centers[cluster_labels]
            x_k = KM.transform(X)                   
            intracluster_d = L2norm(x_dk)
            all_labels['L2 from centroid for k = ' + str(k)] = intracluster_d
            intercenter_d = cdist(cluster_centers, cluster_centers, 'euclidean')
            intercluster_d = L2norm(x_k)
            
            
            # WCSS Scatter Matrix
            WCSM = np.zeros(shape=(ndim,ndim))
            for xidx in range(0,nsamp):
                WCSM += np.mat(x_dk[xidx]).T * np.mat(x_dk[xidx])
                
            # BCSS Scatter Matrix
            BCSM = np.zeros(shape=(ndim,ndim))
            mu = np.mean(X, axis = 0)
            for cidx in range(0,k):
                BCSM += sum(cluster_labels==cidx) * np.mat(cluster_centers[cidx]-mu).T * np.mat(cluster_centers[cidx]-mu)
            
            
            
            ### inertia            
            all_kmetric['inertia'].loc[k] = KM.inertia_
            
            ### Dunn
            all_kmetric['dunn'].loc[k] = np.sqrt(intercenter_d[intercenter_d!=0].min()) / np.sqrt(intracluster_d.max())
            
            ### Davies-Bouldine
            R = np.zeros(shape=(k,))
            for ci in range(0,k):
                for cj in range(0,k):
                    S_i = intracluster_d[cluster_labels==ci].mean()
                    S_j = intracluster_d[cluster_labels==cj].mean()
                    R_ij = np.zeros(shape=(k,k))
                    R_ij[ci,cj] = (S_i + S_j) / intercenter_d[ci,cj]
                    R[ci] = R_ij[R_ij != np.inf].max()
            all_kmetric['db'].loc[k] = 1 / float(k) * R.sum()
            
            
            ### ch            
            all_kmetric['ch'].loc[k] = np.trace(BCSM) / np.trace(WCSM) * float((nsamp - k) / (k - 1))
            
            ### ss            
            all_kmetric['ss'].loc[k] = np.trace(WCSM) / np.trace(BCSM) * float(k)
            
            """
            The silhouette_score gives the average value for all the samples.
            This gives a perspective into the density and separation of the formed clusters
            
            """
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For k =", k,
                "The average silhouette score is :", silhouette_avg)
            all_kmetric['silhouette'].loc[k] = silhouette_avg
            
            
            """
            In the following code, the silhouette scores for *each* sample are calculated and plotted
            # The silhouette value ranges from -1 to +1. A high silhouette value indicates that i is well-matched to its own cluster, 
                and poorly-matched to neighboring clusters. If most points have a high silhouette value, then the clustering 
                solution is appropriate.
            """

            sample_silhouette_values = silhouette_samples(X, cluster_labels)
            all_labels['Silhouette coefficient for k = ' + str(k)] = sample_silhouette_values

            # Create a subplot with 1 row and 2 columns
            fig, (ax1) = plt.subplots(1, 1)
            fig.set_size_inches(10, 8)

            ax1.set_xlim([-0.2, 1])
            # The (k+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (k + 1) * 10])
            
            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values =                     sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.spectral(float(i) / k)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.1, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("Silhouette for k = " + str(k) + "\n(average silhouette coefficient dashed red line)")
            ax1.set_xlabel("Silhouette Coefficient")
            ax1.set_ylabel("Cluster")

            # The vertical line for average silhoutte score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            
            #plt.show()
            fig.savefig(fname+rname + " " + dtname + " silhouette " + str(k) + ".png")
            plt.close("all")
            
            ### Scatter Matrix Plot
            if True:
                    for cidx in range(0,k,1):
                        # plot = df.plot()
                        # fig = plot.get_figure()
                        scatterplot = scatter_matrix(df[cluster_labels==cidx], alpha=0.2, figsize=(10, 8), diagonal='hist')
                        fig = plt.gcf()
                        for plot in range(0,len(scatterplot)):
                            for subplot in range(0,len(scatterplot[plot]),1):
                                ax = scatterplot[plot][subplot]
                                ax.yaxis.label.set_rotation(30)
                                ax.yaxis.label.set_size(8)
                                ax.yaxis.set_ticks([])
                                ax.xaxis.label.set_rotation(30)
                                ax.xaxis.label.set_size(8)
                                ax.xaxis.set_ticks([])
                        fig.suptitle('Scatter Matrix for ' + fname+rname + ' cluster ' + str(cidx) + '(k=' + str(k) + ')')
                        fig.savefig(fname+rname + " " + dtname + " scatter " + str(k) + "-" + str(cidx) +  ".png")
                        plt.close("all")

        all_labels = all_labels.set_index('CIRCUIT')
        df_complete = pd.concat([df,all_labels],axis=1)
        df_complete.to_csv( fname+rname + " " + dtname + " data.csv")
        all_kmetric.to_csv( fname+rname + " " + dtname + " metrics.csv")

