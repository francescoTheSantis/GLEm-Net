from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from math import floor
import os
import matplotlib.pyplot as plt


class GlemNet:

    def __init__(self, df, categorical_cols, label, task, device):
        self.df = df
        self.categorical_cols = categorical_cols
        self.numerical_cols = [x for x in df.columns if x not in categorical_cols+[label]]
        self.label = label
        self.train = None
        self.val = None
        self.test = None
        self.task = task
        self.argument_check()
        self.loaded_train = None
        self.loaded_val = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.cat_cardinality = None
        self.loss = None
        self.tuples = None
        self.feature_names = self.categorical_cols + self.numerical_cols
        self.device = device
        self.eps = None

    def argument_check(self):

        if self.task not in ['Regression', 'Classification']:
            raise ValueError("This methodology is meant to solve only classification or regression problems!")

        if not isinstance(self.categorical_cols, list):
            raise TypeError("Categorical_cols must be a list of strings!")

        for col in self.categorical_cols:
            if not isinstance(col, str):
                raise TypeError("Categorical_cols must be a list of strings containing the names of the columns for which the data are categorical!")
            if col not in self.df.columns:
                raise ValueError("There is a mismatch between the categorical columns provided and the columns of the dataset!")

        if not isinstance(self.label, str):
            raise TypeError("The label must be a string!")

        if self.label not in self.df.columns:
            raise ValueError("The label should be a string representing the name of the independent variable, which must be contained in the dataset provided|")

        if not isinstance(self.df, pd.core.frame.DataFrame):
            raise TypeError("The dataset provided is not a Pandas DataFrame!")

    # preprocessing and train/val/test split
    def preprocessing(self, numerical_preprocessing = 'standardize', categorical_preprocessing = 'ordinal_encoder', splits = [0.7, 0.1, 0.2], random_state=42):

        # params check
        if numerical_preprocessing not in ['standardize','normalize']:
            raise ValueError('numerical_preprocessing can be either standardize or normalize!')

        if categorical_preprocessing not in ['ordinal_encoder','one_hot_encoder']:
            raise ValueError('categorical_preprocessing can be either ordinal_encoder or one_hot_encoder!')

        # a list containing the cardinality of each categorical variable is created
        cardinality_cat_list = []
        for col in self.df.columns:
            if col in self.categorical_cols:
                cardinality_cat_list.append(len(self.df[col].unique()))

        self.cat_cardinality = cardinality_cat_list.copy()

        # the columns of the df are reordered in order to have the categorical
        # columns first and then the numerical ones.
        # this operation makes it easier to apply grouped lasso technique in the pytorch model
        new_order = self.categorical_cols + [x for x in self.df.columns if x not in self.categorical_cols]
        self.df = self.df[new_order]

        # apply if the categorical columns have to one hot encoded
        if categorical_preprocessing == 'one_hot_encoder':
            ohe = OneHotEncoder()
            one_hot_mat = ohe.fit_transform(self.df[self.categorical_cols]).todense()
            one_hot_df = pd.DataFrame(one_hot_mat)
            self.df = pd.concat([one_hot_df.reset_index(), self.df[self.numerical_cols + [self.label]].reset_index()],
                                axis=1).drop(columns=['index', 'index'])

        elif categorical_preprocessing == 'ordinal_encoder':
            ord_encoder = OrdinalEncoder()
            for col in self.categorical_cols:
                if col in self.categorical_cols:
                    self.df[col] = ord_encoder.fit_transform(self.df[col].to_numpy().reshape(-1, 1)).squeeze()

        # split the data into train/val/test
        train_val, test = train_test_split(self.df, test_size=splits[-1],
                                           random_state=random_state, stratify=self.df[self.label])

        if splits[-2] > 0:
            train, val = train_test_split(train_val, test_size=splits[-2],
                                          random_state=random_state, stratify=train_val[self.label])
        else:
            train = train_val.copy()
            val = None

        # set the numerical preprocessing defined by the user: standardization, normalization or None
        if numerical_preprocessing == 'standardize':
            scaler = StandardScaler()
        elif numerical_preprocessing == 'normalize':
            scaler = MinMaxScaler()

        # the scalers are fit using the training set and then used to transform all the different splits
        for col in self.numerical_cols:
            if col in self.numerical_cols:
                train[col] = scaler.fit_transform(train[col].to_numpy().reshape(-1, 1))
                if isinstance(val, pd.DataFrame):
                    val[col] = scaler.transform(val[col].to_numpy().reshape(-1, 1))
                test[col] = scaler.transform(test[col].to_numpy().reshape(-1, 1))

        # The training, validation and test are updated
        self.train = train
        self.val = val
        self.test = test

    def load_data(self, data, batch_size, shuffle):
        x = data.drop(columns=self.label)
        x = torch.tensor(x.to_numpy(), dtype=torch.float32)
        y = data[self.label]
        y = torch.tensor(y.to_numpy(), dtype=torch.float32).view(-1, 1)
        loaded_data = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=batch_size, shuffle=shuffle)
        return loaded_data

    def load_splits(self, batch_size, shuffle):
        self.loaded_train = self.load_data(self.train, batch_size, shuffle)
        if isinstance(self.val, pd.DataFrame):
            self.loaded_val = self.load_data(self.val, batch_size, False)

        self.X_test = self.test.drop(columns=self.label).to_numpy()
        self.y_test = self.test[self.label].to_numpy()

    def create_model(self, neurons_per_layer, hidden_activation='tanh', output_activation='sigmoid', card_reduction=2):
        self.model = MLP_embeddings(neurons_per_layer, self.cat_cardinality, len(self.numerical_cols),
                                    hidden_activation, output_activation, card_reduction, self.device).to(self.device)
        self.tuples = self.model.get_tuples()

    def set_loss(self, lambda_coeff=0, function_dict={'name': 'linear'}, weights=[1, 1]):
        self.loss = GroupedLassoPenalty(lambda_coeff, self.tuples, self.model, function_dict, weights, self.task)

    # Define the training loop
    def train_model(self, num_epochs, folder_path, optimizer, scheduler, verbose=0, freeze=True, eps=1e-3):

        self.eps = eps
        train_losses = []
        val_losses = []
        norms_list = []
        min_loss = np.finfo(np.float64).max
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
                for tup in self.tuples:
                    cat_vec_norms = 0
                    for i in range(tup[0], tup[1]):
                        cat_vec_norms += torch.norm(self.model.layers[0].weight[:, i], p=2)
                    cat_vec_norms = cat_vec_norms / (tup[1] - tup[0])

                    if cat_vec_norms < eps:
                        for i in range(tup[0], tup[1]):
                            to_freeze.append(i)

                # now the rest of the continuous variables
                for i in range(self.tuples[-1][-1], self.model.layers[0].weight.size()[1]):
                    if torch.norm(self.model.layers[0].weight[:, i], p=2) < eps:
                        to_freeze.append(i)

            to_freeze = list(set(to_freeze))

            if verbose == 0:
                print('Freezed norms:', to_freeze)

            norms, _ = self.model.compute_norms()
            norms_list.append(norms)

            for inputs, labels in self.loaded_train:

                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                loss = self.loss(outputs, labels)

                # Backward pass and optimization
                loss.backward()

                if freeze:
                    for i in to_freeze:
                        self.model.layers[0].weight.grad[:, i] = \
                            torch.zeros_like(self.model.layers[0].weight.grad[:, i])

                optimizer.step()

                running_loss += loss.item()

            scheduler.step()

            # Calculate average training loss for the epoch
            train_loss = running_loss / len(self.loaded_train)
            train_losses.append(train_loss)

            if isinstance(self.val, pd.DataFrame):
                val_loss = self.evaluate_model(self.model, self.loaded_val, self.loss, self.device)
                val_losses.append(val_loss)
                if val_loss < min_loss:
                    torch.save(self.model, folder_path + '/best_model.pt')

                if verbose > 0:
                    if epoch % verbose == 0:
                        print('Learning_rate:', optimizer.param_groups[0]["lr"])
                        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch + 1,
                                                                                           num_epochs,
                                                                                           train_loss, val_loss))
                        print()
            else:
                if train_loss < min_loss:
                    torch.save(self.model, folder_path + '/best_model_WholeTraining.pt')

                if verbose > 0:
                    if epoch % verbose == 0:
                        print('Learning_rate:', optimizer.param_groups[0]["lr"])
                        print('Epoch [{}/{}], Train Loss: {:.4f},'.format(epoch + 1, num_epochs, train_loss))
                        print()

        np_norms = np.array(norms_list)
        epochs = np.arange(np_norms.shape[0]) + 1

        # Plot the training and validation loss
        if verbose > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.set_size_inches(15, 10)
            ax1.plot(epochs, train_losses, label='Training Loss')

            if isinstance(self.val, pd.DataFrame):
                ax1.plot(epochs, val_losses, label='Validation Loss')

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid()
            ax1.set_title('Training and Validation Loss during training')
            ax1.legend()

            for var in range(np_norms.shape[1]):
                ax2.plot(epochs, np_norms[:, var], label=self.feature_names[var])

            ax2.grid()
            ax2.set_title('Input vector norms during training')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('||v_i||_2')
            ax2.legend()
            plt.savefig(folder_path + '/norms_vs_losses.png', bbox_inches='tight')

        if isinstance(self.val, pd.DataFrame):
            columns = ['epochs', 'training loss', 'validation loss']
        else:
            columns = ['epochs', 'training loss']

        training_info = pd.DataFrame(columns=columns)

        training_info['epochs'] = epochs
        training_info['training loss'] = train_losses

        if isinstance(self.val, pd.DataFrame):
            training_info['validation loss'] = val_losses

        training_info.to_csv(folder_path+'/training_results.csv')

        norms_info = pd.DataFrame(np_norms)
        norms_info.columns = self.feature_names
        norms_info['epochs'] = list(range(1, num_epochs+1))

        norms_info.to_csv(folder_path+'/norms.csv')

    def evaluate_model(self, model, data_loader, criterion, device):
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

    def plot_features_usage(self):
        title = 'norms'
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        norms = self.model.compute_norms()[0]
        # Sort feature importances in descending order
        indices = np.argsort(norms)[::-1]
        sorted_feature_importances = np.array(norms)[indices]
        sorted_feature_names = np.array(self.feature_names)[indices]

        colors = []
        for value in sorted_feature_importances:
            if value > self.eps:
                colors.append('tab:blue')
            else:
                colors.append('tab:red')

        ax.bar(range(len(sorted_feature_importances)), sorted_feature_importances, tick_label=sorted_feature_names,
               color=colors)
        ax.axhline(y=self.eps, color='r', linestyle='-', label='threshold: '+str(self.eps))
        ax.set_xlabel('Feature')
        ax.set_ylabel(title)
        ax.grid()
        ax.legend()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        plt.tight_layout()
        plt.show()

    def predict(self):
        return self.model(torch.Tensor(self.X_test).to(self.device)).detach().squeeze().cpu().numpy()

    def prune(self):

        to_zero=[]
        # add the index of the categorical variable weights which need to be zeroed
        for tup in self.tuples:
            cat_vec_norms = 0
            for i in range(tup[0], tup[1]):
                cat_vec_norms += torch.norm(self.model.layers[0].weight[:, i], p=2)
            cat_vec_norms = cat_vec_norms / (tup[1] - tup[0])

            if cat_vec_norms < self.eps:
                for i in range(tup[0], tup[1]):
                    to_zero.append(i)

        # now the numerical variables
        for i in range(self.tuples[-1][-1], self.model.layers[0].weight.size()[1]):
            if torch.norm(self.model.layers[0].weight[:, i], p=2) < self.eps:
                to_zero.append(i)

        with torch.no_grad():
            for i in to_zero:
                self.model.layers[0].weight[:, i] = torch.zeros_like(self.model.layers[0].weight[:, i])


# Class which defines the model
class MLP_embeddings(nn.Module):

    # specify:
    #   - a list containing the number of neurons for each layer
    #   - a list containing the positions (columns) of the categorical variables in the input matrix
    #   - a list containing the cardinality of each variable
    #   - activation function to be used by the mlp
    #   - activation function to be used by the output layer (to be chosen according to the problem)

    def __init__(self, neurons_per_layer, cat_cardinality, n_numerics, hidden_activation='tanh',
                 output_activation='sigmoid', card_reduction=0, device='cpu'):
        super(MLP_embeddings, self).__init__()

        # create multiple embeddings according to the number of categorical variables
        self.cat_cardinality = cat_cardinality
        self.neurons_per_layer = neurons_per_layer
        self.n_cats = len(cat_cardinality)
        self.n_numerics = n_numerics
        self.card_reduction = card_reduction
        if card_reduction > 0:
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
            params_threshold = (card * self.neurons_per_layer[0]) / (card + self.neurons_per_layer[0])
            embedding_dim = floor(params_threshold / card_reduction)
            if embedding_dim == 0:
                neurons_per_embedding.append(1)
                embeddings.append(nn.Embedding(card, 1))
            else:
                neurons_per_embedding.append(embedding_dim)
                embeddings.append(nn.Embedding(card, embedding_dim))
        return embeddings, neurons_per_embedding

    def set_act(self, act_name):
        if act_name == 'relu':
            act = nn.ReLU()
        elif act_name == 'tanh':
            act = nn.Tanh()
        elif act_name == 'sigmoid':
            act = nn.Sigmoid()
        else:
            raise ValueError("Activation function not available.")
        return act

    def create_mlp(self):
        layers_list = nn.ModuleList()
        for i in range(len(self.neurons_per_layer)):
            if i == 0:
                if self.card_reduction > 0:
                    layers_list.append(nn.Linear(sum(self.neurons_per_embedding) + self.n_numerics,
                                                 self.neurons_per_layer[i]))
                else:
                    layers_list.append(nn.Linear(sum(self.cat_cardinality) + self.n_numerics,
                                                 self.neurons_per_layer[i]))
            else:
                layers_list.append(nn.Linear(self.neurons_per_layer[i - 1], self.neurons_per_layer[i]))
        return layers_list

    def get_tuples(self):
        tuples = []
        idx = 0

        if self.card_reduction > 0:
            cardinalities = self.neurons_per_embedding.copy()
        else:
            cardinalities = self.cat_cardinality.copy()

        for i in range(len(cardinalities)):
            tuples.append([idx, idx + cardinalities[i]])
            idx = sum(cardinalities[:i + 1])
        return tuples

    def compute_norms(self):
        norms = []

        for tup in self.tuples:
            cat_vec_norms = 0
            for i in range(tup[0], tup[1]):
                cat_vec_norms += torch.norm(self.layers[0].weight[:, i], p=2)
            cat_vec_norms = float(cat_vec_norms) / (tup[1] - tup[0])
            norms.append(cat_vec_norms)

        for i in range(self.tuples[-1][-1], self.layers[0].weight.size()[1]):
            norms.append(float(torch.norm(self.layers[0].weight[:, i], p=2)))

        perc_norms = 100 * np.array(norms) / np.sum(norms)
        return norms, perc_norms

    def forward(self, x):

        if self.card_reduction > 0:

            # slice the input matrix in order to give to each embedding the associated categorical variable
            embedded_list = [self.embeddings[i](x[:, i].type(torch.IntTensor).to(self.device)) for i in range(self.n_cats)]
            out = torch.cat((*embedded_list, x[:, self.n_cats:]), dim=1)
        else:
            out = x

        for layer in self.layers:
            out = layer(out)
            out = self.hidden_activation(out)

        out = self.fc(out)
        out = self.output_activation(out)
        return out


# Negative loglikelihood with input weights penalty
class GroupedLassoPenalty(nn.Module):

    def __init__(self, lambda_coeff, tuples, model, function_dict, weights, task):
        super(GroupedLassoPenalty, self).__init__()

        self.task = task
        self.eps = torch.finfo(torch.float32).eps
        self.lambda_coeff = lambda_coeff
        self.tuples = tuples
        self.model = model
        self.function = self.set_function(function_dict)
        self.weights = weights

    def set_function(self, function_dict):

        name = function_dict['name']

        if name == 'quadratic':
            def quadratic(x, alpha=function_dict['alpha']):
                if torch.abs(x) >= alpha:
                    return x
                else:
                    return ((x ** 2) / (2 * alpha)) + alpha / 2
            return quadratic

        elif name == 'linear':
            def linear(x):
                return x
            return linear

        elif name == 'exp':
            def composed(x, beta=function_dict['beta'], gamma=function_dict['gamma']):
                return 1 - torch.exp(-(x ** 2) / beta) + gamma * torch.abs(x)
            return composed

        elif name == 'tanh':
            def tanh(x, beta=function_dict['beta'], gamma=function_dict['gamma']):
                return torch.tanh(x / beta) + gamma * x
            return tanh

        else:
            raise ValueError("Function not available.")

    def forward(self, p, y):

        if self.task == 'Classification':
            task_loss = -(self.weights[0] * y * torch.log(torch.clamp(p, min=self.eps, max=1 - self.eps)) + self.weights[1] * (1 - y) * torch.log(torch.clamp(1 - p, min=self.eps, max=1 - self.eps)))
        else:
            task_loss = (y-p).pow(2)

        avg_task_loss = torch.mean(task_loss)

        if self.lambda_coeff > 0:
            input_weight_norms = 0

            # first let's consider the categorical variables
            for tup in self.tuples:
                cat_vec_norms = 0
                for i in range(tup[0], tup[1]):
                    cat_vec_norms += torch.norm(self.model.layers[0].weight[:, i], p=2)
                cat_vec_norms = cat_vec_norms / (tup[1] - tup[0])
                input_weight_norms += self.function(cat_vec_norms)

            # now the rest of the continuous variables
            for i in range(self.tuples[-1][-1], self.model.layers[0].weight.size()[1]):
                input_weight_norms += self.function(torch.norm(self.model.layers[0].weight[:, i], p=2))

            loss = avg_task_loss + input_weight_norms * self.lambda_coeff

        else:
            loss = avg_task_loss

        return loss
