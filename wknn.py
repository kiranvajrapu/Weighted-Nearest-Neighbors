#importing packages
import numpy as np
import pandas as pd 
import math #for mmathematical opeartions like sqrt...
from scipy.spatial import distance #for euclidean distance(if don't specify any funtion then use this)
from scipy.stats import norm #for gaussian kernal
import operator #for sorting


#Mean
def mean(x):
    cou=0
    sum=0
    for i in range(len(x)):
        if(math.isnan(x[i])):
            continue
        else:
            sum += x[i]
            cou += 1
    ans=(sum/cou)
    return ans
	
#Standard Deviation
def std(x):
    xm=mean(x)
    cou = 0
    sum = 0
    ans=0
    for i in range(len(x)):
        if(math.isnan(x[i])):
            continue
        else:
            sum += ((x[i] - xm)**2)
            cou += 1
    if(cou>1):
        cou=cou-1
        ans = math.sqrt(sum/cou)
    return ans


#Covariance
def cov(x,y):
    sum=0
    cou =0
    xm=mean(x)
    ym=mean(y)
    for i in range(len(x)):
        if(math.isnan(x[i]) or math.isnan(y[i])):
            continue
        else:
            cou += 1
            sum += (x[i]-xm) * (y[i]-ym)
    if(cou > 1):
        cou = cou-1
        ans = (sum/cou)
    else:
        ans=0
    return ans



#Correlation 
def corr(x,y):
    sx = std(x)
    sy = std(y)
    c = cov(x,y)
    if ( sx == 0 or sy == 0):
        corr = 0
    else:
        corr = (c/(sx * sy))
    return corr



#Euclidean distance funtion
def euclidean_distance(lst1,lst2,w):
    sum=0
    cou=0
    for x in range(len(w)):
        if(w[x]==0):
            continue
        else:
            
            if(math.isnan(lst1[x]) or math.isnan(lst2[x])):
                continue
            else:
                cou += 1
                sum  = sum + (((lst1[x] - lst2[x])**2) * w[x])
    #print(sum)
    if(cou>0):
        sum=math.sqrt(sum/cou)
    else:
        sum=math.sqrt(sum)
    
    return sum
	
def fill_value(distances,k):
    v = 0
    total_weight = 0
    #print(distances)
    for i in range(k):
        weight = norm.pdf(distances[i][1],loc=5,scale=10) #call the built in gaussian kernal functon to calculate the weight
        v += distances[i][0]*weight
        total_weight += weight
    ans=v/total_weight
    return ans
	
	
def wkNN_impute(data):
    """Function that provides imputation by wkNN- takes the gaussian distribution weights of the values for 3,5,10 nearest neighbors (if possible) with valid data 
    
    Parameters:
    -----------
    data: data from which dataset you want to impute
    
    Output:
    -------
    Saves csv matrix "ImputedMatrix.csv" .
    
    Notes:
    ------
    If column is completely empty, drops it 
    """
	#add index column so that original row numbers are maintained
	data['INDEX'] = range(len(data))
	print(data.shape)
	
	#deletes any columns that are empty in dataset
	count=0
	for column in data:
		if pd.isnull(data[column]).all():
			del data[column] 
			count=count+1 
	print(count) #to know how many  columns have null values.
	
	#Dataset can be converted to matix representation for our flexibility.
	matrix = data_test.as_matrix()
	complete_matrix = np.copy(matrix)
	print(matrix)
	
	for x in range(matrix.shape[0]):
		if np.isnan(matrix[x]).any(): #if we need to impute for this row
			
			#figure out which columns have nan's for this row
			indices_to_impute=list(np.transpose(np.argwhere(np.isnan(matrix[x])))[0])
			
			######
            #In terms of wkNN input, drop columns that are missing for this example, drop the test example row 
            #itself, and drop any rows that still have missing data
            ######
			
			#We calculate first correlation between imputed row and other rows
			test=pd.DataFrame(matrix)
			test=pd.DataFrame(test,columns=indices_to_impute)
			test_corr=test.drop(test.index[[x]]) #Checking correlation for null to other columns, so we store columns contain null values for particular row.
			training_X = np.delete(matrix, indices_to_impute, axis=1) #drop columns that are missing for this example
			# for our convinience which value impute in particular index(row) that can be store at test_example and training_X is remaining rows to find out neighbours
			test_example = training_X[x]
			training_X = np.delete(training_X, x, axis = 0) #drop the test example
			training_X = training_X[~np.isnan(training_X).any(axis=1)] #drop the rows with missing data
       
			#model inputs- ignore last "Index" Column 
			model_X = training_X[:,:-1]
			model_Y = test_example[:-1]
			
			df=pd.DataFrame(training_X)
			lst=df[df.columns[len(df.columns)-1]] #for using last index column which rows are neighbous that can be stored at list
			
			train_corr=pd.DataFrame(model_X)
			#iterate over each entry in row that needs imputing
			for index in indices_to_impute:
				i=0
				dist=[]
				weight=[]
				for column in train_corr:	#Correlation can be done with index of column contain null value to other columns.
					x_corr=list(test_corr[index])
					y_corr=list(train_corr[column])
					r=corr(x_corr,y_corr)
					r=abs(r)
					weight.append(r ** 2)
				for i in range(mat1.shape[0]):
					dis=euclidean_distance(model_Y,model_X[i],weight) #call the  function euclidean_distance
					true=int(lst[i]) #recover "true" index (before things were deleted)
					if not np.isnan(matrix[true,index]):
						dist.append((matrix[true,index],dis)) #distances are appended according to the value
					
				dist.sort(key=operator.itemgetter(1)) #distances can be sorted fo find nearest by K value
				#fill with gaussian distribution of neighbors if numerical column
				complete_matrix[x,index]=fill_value(dist,k=5) #k=3,5,10 to check the peerformance
				
				
	print(complete_matrix) #print the total imputed matrix.
	
	#To check whether any null values  are present or not after imputation.
	for x in range(complete_matrix.shape[0]):
    if np.isnan(complete_matrix[x]).any():
        print(x)
        print(list(np.transpose(np.argwhere(np.isnan(complete_matrix[x])))[0]))
		
	#convert (complete_matrix) data to DataFrame, delete Index column and Save to csv file
	data = pd.DataFrame(data=complete_matrix[:,:],columns=list(data_impute))  
    
    del data['INDEX']
	data.to_csv("ImputedMatrix.csv", index = False)


	
