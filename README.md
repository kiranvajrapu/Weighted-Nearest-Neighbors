# Weighted-Nearest-Neighbors
Function that provides imputation by wkNN- takes the gaussian distribution weights of the values for 3,5,10 nearest neighbors (if possible) with valid data 
    
    Parameters:
    -----------
    data: data from which dataset you want to impute
    
    Output:
    -------
    Saves csv matrix "ImputedMatrix.csv" .
    
    Notes:
    ------
    If column is completely empty, drops it 
    
