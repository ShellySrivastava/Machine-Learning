import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:, 0] = f1
X_full[:, 1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# X_phonemes_1_2 = ...

# y is the target phoneme value
y = np.array([])
X_phonemes_1_2 = np.empty((0, 2))
for i, phoneme in enumerate(phoneme_id):
    if phoneme == 1 or phoneme == 2:
        X_phonemes_1_2 = np.append(X_phonemes_1_2, [X_full[i]], axis=0)
        y = np.append(y, phoneme)
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

# as dataset X, we will use only the samples of the chosen phoneme
X = X_phonemes_1_2.copy()
# get number of samples
N = X.shape[0]
# change k here
k = 3

GMM_file_01 = 'data/GMM_params_phoneme_01_k_{:02}.npy'.format(k)
# Loading data from .npy file
GMM_parameters_1 = np.load(GMM_file_01, allow_pickle=True)
GMM_parameters_1 = GMM_parameters_1.item()

mu_1 = GMM_parameters_1['mu']
s_1 = GMM_parameters_1['s']
p_1 = GMM_parameters_1['p']

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z_1 = np.zeros((N,k))

Z_1 = get_predictions(mu_1, s_1, p_1, X)
sum1 = Z_1.sum(axis=1)

GMM_file_02 = 'data/GMM_params_phoneme_02_k_{:02}.npy'.format(k)
# Loading data from .npy file
GMM_parameters_2 = np.load(GMM_file_02, allow_pickle=True)
GMM_parameters_2 = GMM_parameters_2.item()

mu_2 = GMM_parameters_2['mu']
s_2 = GMM_parameters_2['s']
p_2 = GMM_parameters_2['p']

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z_2 = np.zeros((N,k))

Z_2 = get_predictions(mu_2, s_2, p_2, X)
sum2 = Z_2.sum(axis=1)

prediction = np.array([])

for i in range(y.size):
    if sum1[i] > sum2[i]:
        prediction = np.append(prediction, 1)
    else:
        prediction = np.append(prediction, 2)

accuracy = (sum(np.equal(prediction, y)) / y.size) * 100

########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()