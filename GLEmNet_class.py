import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from math import floor
from sklearn import preprocessing
import torchvision
# from torchview import draw_graph
from sklearn.ensemble import RandomForestClassifier
from math import floor
from sklearn.model_selection import GridSearchCV
import os
from torch.optim.lr_scheduler import StepLR


# Define the MLP network
# num_cat_classes = list of int representing the cardinality of each categorical variable

class MLP_embeddings(nn.Module):

    # specify:
    #   - a list containing the number of neurons for each layer
    #   - a list containing the positions (columns) of the categorical variables in the input matrix
    #   - a list containing the cardinality of each variable
    #   - activation function to be used by the mlp
    #   - activation function to be used by the output layer (to be chosen according to the problem)

    def __init__(self, neurons_per_layer, cat_cardinality, n_numerics, hidden_activation='tanh', output_activation='sigmoid', card_reduction=2, device='cuda'):
        super(MLP_embeddings, self).__init__()

        # create multiple embeddings according to the number of categorical variables
        self.cat_cardinality = cat_cardinality
        self.neurons_per_layer = neurons_per_layer
        self.n_cats = len(cat_cardinality)
        self.n_numerics = n_numerics

        self.embeddings, self.neurons_per_embedding = self.get_embeddings(card_reduction)
        self.hidden_activation = self.set_act(hidden_activation)
        self.output_activation = self.set_act(output_activation)

        # specify the dimensions of the mlp
        self.fc = nn.Linear(neurons_per_layer[-1], 1)
        self.layers = self.create_mlp()
        self.tuples = self.get_tuples()
        
        self.device = device
        
    def get_embeddings(self, card_reduction):
        neurons_per_embedding = []
        embeddings = nn.ModuleList()
        for card in self.cat_cardinality:
            params_threshold = (card*self.neurons_per_layer[0])/(card + self.neurons_per_layer[0])
            embedding_dim = floor(params_threshold/card_reduction)
            if embedding_dim==0:
                neurons_per_embedding.append(1)
                embeddings.append(nn.Embedding(card, 1))
            else:
                neurons_per_embedding.append(embedding_dim)
                embeddings.append(nn.Embedding(card, embedding_dim))
        return embeddings, neurons_per_embedding

    def set_act(self, act_name):
        if act_name=='relu':
            act=nn.ReLU()
        elif act_name=='tanh':
            act=nn.Tanh()
        elif act_name=='sigmoid':
            act=nn.Sigmoid()
        else:
            raise ValueError("Activation function not available.")
        return act

    def create_mlp(self):
        layers_list = nn.ModuleList()
        for i in range(len(self.neurons_per_layer)):
            if i==0:
                layers_list.append(nn.Linear(sum(self.neurons_per_embedding) + self.n_numerics, self.neurons_per_layer[i]))
            else:
                layers_list.append(nn.Linear(self.neurons_per_layer[i-1], self.neurons_per_layer[i]))
        return layers_list
    
    def get_tuples(self):
        tuples = []
        idx = 0
        embedded_cardinalities = self.neurons_per_embedding.copy()
        for i in range(len(embedded_cardinalities)):
            tuples.append([idx, idx + embedded_cardinalities[i]])
            idx = sum(embedded_cardinalities[:i+1])
        return tuples

    def compute_norms(self):
        norms = []

        for tup in self.tuples:
            # input_weight_norms+= custom_function(torch.norm(torch.flatten(self.model.layers[0].weight[:, tup[0]:tup[1]]), p=self.norm) / (tup[1]-tup[0]))
            cat_vec_norms = 0
            for i in range(tup[0],tup[1]):
                cat_vec_norms += torch.norm(self.layers[0].weight[:,i], p=2)
            cat_vec_norms = float(cat_vec_norms) / (tup[1]-tup[0])
            norms.append(cat_vec_norms)

        for i in range(self.tuples[-1][-1], self.layers[0].weight.size()[1]):
            # norms.append(np.linalg.norm(self.layers[0].weight.detach().cpu().numpy()[:, i], ord=2))
            norms.append(float(torch.norm(self.layers[0].weight[:,i], p=2)))

        perc_norms = 100 * np.array(norms) / np.sum(norms)
        return norms, perc_norms

    def forward(self, x):

        # slice the input matrix in order to give to each embedding the associated categorical variable
        embedded_list = [self.embeddings[i](x[:,i].type(torch.IntTensor).to(self.device)) for i in range(self.n_cats)]

        out = torch.cat((*embedded_list, x[:,self.n_cats:]), dim=1)

        for layer in self.layers:
            out = layer(out)
            out = self.hidden_activation(out)

        out = self.fc(out)
        out = self.output_activation(out)

        return out    

    
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Negative loglikelihood with input weights penalty
class LNLL_emb(nn.Module):
    
    def __init__(self, lambda_coeff, tuples, model, function_dict, weights):
        super(LNLL_emb, self).__init__()
        
        self.eps = torch.finfo(torch.float32).eps
        self.lambda_coeff = lambda_coeff
        self.tuples = tuples
        self.model = model
        self.function = self.set_function(function_dict)
        self.weights = weights

    def set_function(self, function_dict):
        
        name = function_dict['name']

        if name=='quadratic':
            def quadratic(x, alpha=function_dict['alpha']):
                if torch.abs(x)>=alpha:
                    return x
                else:
                    return ((x**2)/(2*alpha)) + alpha/2
            return quadratic

        elif name=='exp':
            def exp(x, beta=function_dict['beta']):
                return 1-torch.exp(-(x**2)/beta) + torch.abs(x)            
            return exp
    
        elif name=='linear':
            def linear(x):
                return x
            return linear 
        
        elif name=='composed':
            def composed(x, beta = function_dict['beta'], gamma = function_dict['gamma']):
                return 1 - torch.exp(-(x**2)/beta) + gamma*torch.abs(x)
            return composed
            
        elif name=='tanh':
            def tanh(x, beta = function_dict['beta'], gamma = function_dict['gamma']):
                return torch.tanh(x/beta) + gamma*x
            return tanh
            
        else:
            raise ValueError("Function not available.")


    def forward(self, p, y):
        negative_log_likelihood_sum = -( self.weights[0] * y * torch.log(torch.clamp(p, min=self.eps, max=1-self.eps)) + 
                                         self.weights[1] * (1 - y) * torch.log(torch.clamp(1 - p, min=self.eps, max=1-self.eps)) )
        negative_log_likelihood_loss = torch.mean(negative_log_likelihood_sum)
    
        input_weight_norms=0

        # first let's consider the categorical variables
        for tup in self.tuples:
            cat_vec_norms = 0
            for i in range(tup[0],tup[1]):
                cat_vec_norms += torch.norm(self.model.layers[0].weight[:,i], p=2)
            cat_vec_norms = cat_vec_norms / (tup[1]-tup[0])
            input_weight_norms += self.function(cat_vec_norms)

        # now the rest of the continuous variables
        for i in range(self.tuples[-1][-1], self.model.layers[0].weight.size()[1]):
            input_weight_norms+= self.function(torch.norm(self.model.layers[0].weight[:,i], p=2))

        Loss = negative_log_likelihood_loss + input_weight_norms * self.lambda_coeff
        return Loss

      


# Negative loglikelihood with input weights penalty
class NLL(nn.Module):
    
    def __init__(self, weights):
        super(NLL, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.weights = weights

    def forward(self, p, y):
        negative_log_likelihood_sum = -( self.weights[0] * y * torch.log(torch.clamp(p, min=self.eps, max=1-self.eps)) + 
                                         self.weights[1] * (1 - y) * torch.log(torch.clamp(1 - p, min=self.eps, max=1-self.eps)) )
        return torch.mean(negative_log_likelihood_sum)
    
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    
    
# Split and load training and test set
def load_data(X, y, batch_size, stratified):
    if stratified==True:
        class_weights = pd.Series(y.squeeze().numpy().astype(int)).value_counts().to_dict()
        weights = []
        weight_1 = 1
        weight_0 = 2
        for label in y.squeeze().numpy():
            if label.astype(int)==1:
                weights.append(weight_1)
            else:
                weights.append(weight_0)
        # weights = torch.Tensor(weights).view(-1,1)
        sampler = WeightedRandomSampler(weights, len(y_train)) 
        loaded_data = torch.utils.data.DataLoader(list(zip(X, y)), sampler=sampler, batch_size=batch_size)
    else:
        loaded_data = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=batch_size, shuffle=True)
    return loaded_data




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------




def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    model.train()
    return val_loss / len(data_loader)



# Define the training loop
def train_mlp(model, tuples, train_loader, val_loader, num_epochs, device, feature_names, loss_function, folder_name, freeze, eps, optimizer, scheduler):
    
    criterion = loss_function

    train_losses = []
    val_losses = []
    norms_list = []

    min_loss = np.finfo(np.float64).max
    
    folder_path = folder_name

    to_freeze = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
    
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        
        if freeze:
            # first let's consider the categorical variables
            for tup in tuples:
                cat_vec_norms = 0
                for i in range(tup[0],tup[1]):
                    cat_vec_norms += torch.norm(model.layers[0].weight[:,i], p=2)
                cat_vec_norms = cat_vec_norms / (tup[1]-tup[0])

                if cat_vec_norms<eps:
                    for i in range(tup[0],tup[1]):
                        to_freeze.append(i)

            # now the rest of the continuous variables
            for i in range(tuples[-1][-1], model.layers[0].weight.size()[1]):
                if torch.norm(model.layers[0].weight[:,i], p=2)<eps:
                    to_freeze.append(i)
        
        to_freeze = list(set(to_freeze))
        
        print('Freezed norms:', to_freeze)
        
        norms, _ = model.compute_norms()
        norms_list.append(norms)

#         if epoch in decay:
#               learning_rate = learning_rate*0.1

        for inputs, labels in train_loader:

            labels = labels.to(device)
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            
            if freeze:
                for i in to_freeze:
                    model.layers[0].weight.grad[:,i] = torch.zeros_like(model.layers[0].weight.grad[:,i])
            
            
            optimizer.step()

            running_loss += loss.item()

        
        scheduler.step()
        
        # Calculate average training loss for the epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        if val_loader!=None:
            val_loss = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            if val_loss < min_loss:
                torch.save(model, folder_path+'/best_model.pt')
            print('Learning_rate:', optimizer.param_groups[0]["lr"])
            print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss))
        else: 
            if train_loss < min_loss:
                torch.save(model, folder_path+'/best_model_WholeTraining.pt')
            print('Learning_rate:', optimizer.param_groups[0]["lr"])
            print('Epoch [{}/{}], Train Loss: {:.4f},'.format(epoch+1, num_epochs, train_loss))

        
    np_norms = np.array(norms_list)
    epochs = np.arange(np_norms.shape[0])+1
        
    if val_loader!=None:
        # Plot the training and validation loss
        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.set_size_inches(15,10)
        ax1.plot(epochs, train_losses, label='Training Loss')
        ax1.plot(epochs, val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid()
        ax1.set_title('Training and Validation Loss during training')
        ax1.legend()
    
    if val_loader==None:
        val_losses=None
        
        # Plot the training and validation loss
        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.set_size_inches(15,10)
        ax1.plot(epochs, train_losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid()
        ax1.set_title('Training and Validation Loss during training')
        ax1.legend()

    for var in range(np_norms.shape[1]):
        ax2.plot(epochs, np_norms[:,var], label = feature_names[var])
        
    ax2.grid()
    ax2.set_title('Input vector norms during training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('||v_i||_2')    
    plt.savefig(folder_path+'/norms_vs_losses.png', bbox_inches='tight')
    return train_losses, val_losses, np_norms, epochs


def plot_features_usage(mlp, feature_names, norms, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(10,5)

    # Sort feature importances in descending order
    indices = np.argsort(norms)[::-1]
    sorted_feature_importances = np.array(norms)[indices]
    sorted_feature_names = np.array(feature_names)[indices]
    ax.bar(range(len(sorted_feature_importances)), sorted_feature_importances, tick_label=sorted_feature_names)
    ax.set_xlabel('Feature')
    ax.set_ylabel(title)
    ax.grid()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    plt.tight_layout()
    plt.show()


