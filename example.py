# import 
from GLEm_Net import GlemNet
from torchview import draw_graph
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch

# load the dataset
data = pd.read_csv('Datasets/Sleep_health_and_lifestyle_dataset.csv', delimiter=',').dropna().drop(columns='Person ID')

# data cleaninig
data['Sleep Disorder'] = data.apply(lambda row: 0 if row['Sleep Disorder'] == 'Sleep Apnea' else 1, axis=1)
data['blood Pressure max'] = data.apply(lambda row: int(row['Blood Pressure'].split('/')[0]), axis=1)
data['blood Pressure min'] = data.apply(lambda row: int(row['Blood Pressure'].split('/')[1]), axis=1)
data = data.drop(columns='Blood Pressure')

# store in a list the name of the categorical variables
categorical = []
for col, ty in zip(data.dtypes.index, data.dtypes):
    if ty=='object':
        categorical.append(col)

# define the task the we want to perform (classification or regression)
task = 'Classification'

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the glem_net class specifying the list of categorical and numerical columns, 
# and the label name.
# Additionally, also the task and device are specified
glem_class = GlemNet(data, categorical, 'Sleep Disorder', task, device)

# Using this class it is possible to split, the data and apply the necessary preprocessing techniques to 
# the categorical and numerical features.
glem_class.preprocessing(numerical_preprocessing = 'standardize', categorical_preprocessing= 'one_hot_encoder', splits = [0.7, 0.1, 0.2], random_state = 42)

# The preivously created split are tranformed to tnesor and loaded for the training.
glem_class.load_splits(30, True)

# Using this function the model having the desired architecure is set. 
glem_class.create_model([16,4], hidden_activation='tanh', output_activation='sigmoid', card_reduction=0)

# Hyper-params required for the training
lambda_coeff = 0.06
function_dict = {'name':'exp', 'beta':1e-3, 'gamma':1}
glem_class.set_loss(lambda_coeff, function_dict)
folder_name = 'results'
learning_rate = 0.01
epochs = 500
verbose = 100
freeze = True
eps = 1e-3

                
# The optimizer and the scheduler have to be set outside of the class
optimizer = optim.SGD(glem_class.model.parameters(), lr=learning_rate, momentum=0, nesterov=False)
scheduler = StepLR(optimizer, step_size=400, gamma=0.5)

# The model is trained and the results are stored in the folder specified by the user
glem_class.train_model(epochs, folder_name, optimizer, scheduler, verbose, freeze, eps)

# This function can show the values of the feature norms at the end of the training
glem_class.plot_features_usage('esempio', eps)

