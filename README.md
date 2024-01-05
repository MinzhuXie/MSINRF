# MSINRF
Predict CircRNA-Disease Associations Based on  Random Walk with Restart, Principal Component Analysis and Random Forest

## Requirement

- Python 3.6

- Numpy

- Sklearn

- scipy

- matplotlib

- MATLAB

### Quick start
We provide an example script to run experiments on our dataset: 

- Run `./python/misf.py`: predict CircRNA-disease interactions, and evaluate the results with five-fold cross-validation. 

### All process
1. -Run `main.m`

2. -Run `run_kfold.m.m`

3. -Run `misf.py`

### Code and data
#### `matlab/` directory
- `main.m`: compute similarity based on interaction/association network
- `run_kfold.m`: combine the feature vector of CircRNA-disease pairs and adopt five-fold cross-validation to divide a dataset 

#### `python/` directory
- `allfiles.py`: load all the dataset 
- `misf.py`: use the dataset to run RandomForest

#### `./smallrawdata1` directory
- `Association matrix.txt`: CircRNA-Disease interaction matrix
- `Similarity_Matrix_Proteins.txt`: Disease similarity scores based on  disease semantic similarity
