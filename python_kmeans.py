import matplotlib.pyplot as plt
import numpy 
import pandas as pd

#read in the iris dataset
data = pd.read_csv(open('./Documents/iris.csv'))
# you can specify what you want to be x and y
x = data['Sepal.Length']
y = data['Petal.Length']
# turn the data into a numpy array
d = [0 for i in range(len(x))]
for j in range(len(x)):
    
    d[j] = [x[j],y[j]]
    
X = numpy.array(d)
    
#
# param: data - the data we want Xto calculate the total within sum of squares
# param: clusters - the cluster assignments for each data point
# param: k - the number of clusters
#
# return: the total within sum of squares
#
def within_sumoSq(data, clusters, k):

    xbar = 0
    inside = [0 for i in range(k)]

    #calculate the mean for each cluster
    #subtract the mean from each point in a cluster
    #square, and add
    for i in range(k):
        dt = data[clusters == i,:]
        xbar = numpy.mean(dt, axis = 0)
        n,m = numpy.shape(dt)
        repeated = 
        inside[i] = 

    return inside

#
# param: data - the data we want the total sum of squares
#
# return: the total sum of squares
#
def tot_sumoSq(data):
    
    #need to calculate the total average for each cluster
    #then create a matrix for both
    avg = 
    n,m = 
    repeated_mean = 
  
    #then do the total sum of squares formula
    return( numpy.sum( (data - repeated_mean) ** 2 ) )

#
# param: data - the data we want to calculate between sum of squares
# param: clusters - the cluster assignments
# param: k - the number of clusters
#
# return: the between cluster sum of squares
#  
def between_sumoSq(data, clusters, k):  
  
  inside = 
  totes = 
  
  return( totes - inside )
 
#
# param: data - what we want to partition
# param: k - how many partitions do we want
# param: maxiter - maximum iterations
#
# return: the k centers
# return: the cluster assignments
# return: within sum of squares
# return: between sum of squares
# return: the F-ratio - between sumoSq divided by total sumoSq
#  
def my_kmeans(data, k, maxiter):
    
    #determine dimensions of data
    numIndiv, numFeatures = np.shape(data)
  
    #initialize the centers to be some random
    #subset of the data
    centers = 
    #initialize a vector for cluster assignments
    clusterAssignments = [ 0 for j in range(numIndiv) ]
    #initialize the distance
    d = [ 0 for j in range(k) ]
  
    #we'll iterate the number of times you deem necessary
    for iter in range(maxiter):
    
        #this loop will iterate over all data points
        #we'll calculate the distance from all centers
        #then assign the data point to the cluster whose
        #center is the smallest distance
        for i in range(numIndiv):
      
            #determine distances to all centers
            for j in range(k):
        
                d[j] = 
        
            #assign the point to the cluster with the 
            #smallest distance
            clusterAssignments[i] = 
        
        
        #now update the new centers by calculating 
        #the new means
        for j in range(k):
            indx = 
            centers[j,:] = numpy.mean(data[indx,:],axis = 0)
    
    #calculate the within sum of squares. 
    #we want points to be more similar within clusters
    #so we expect this to be smaller than between
    withinSS = 
  
    #calculate the between sum of squares.
    #we want points to be less similar between clusters
    #so we expect this to be larger than the within 
    betweenSS = 
  
    totSS = 
  
    #To check effectiveness, look at betweenSS/sum(withinSS)
    #if it's large, then success!  You should be asking, how large
    #is too large
    ratio = 
  
    #return both the cluster assignments and the final centroids
    return {"assign":numpy.array(clusterAssignments), "center":centers, "wss":withinSS,"bss":betweenSS,"F_ratio":ratio}
 
km = my_kmeans(X,3,100)

cluster_assign = km["assign"]
clus1 = X[cluster_assign == 0,:]
clus2 = X[cluster_assign == 1,:]
clus3 = X[cluster_assign == 2,:]

fig1 = plt.figure(1)
plt.scatter(clus1[:,0],clus1[:,1],color = 'red', marker = 'o')
plt.scatter(clus2[:,0],clus2[:,1],color = 'blue', marker = '+')
plt.scatter(clus3[:,0],clus3[:,1],color = 'purple', marker = '*')
plt.title("K-Means and Iris")  
fig1.show()