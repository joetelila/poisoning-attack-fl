import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fclusterdata


class Flutils:
    def __init__(self, device="cpu"):
        self.device = device

    # you have to implement the defense algorithm in here.
    @staticmethod
    def aggregateModel(client_models, global_model, defend=False):
        
        if defend:
            print("Defending against attack")
            safe_clients = Flutils.defendFlipAttack(global_model, client_models)
            client_models = safe_clients
        
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            update = [client_models[i].state_dict()[k].float() for i in range(len(client_models))]
            global_dict[k] = torch.stack(update, axis=0).mean(axis=0)
        return global_dict
    
    # Random Attack
    @staticmethod
    def randomFlipAttack(inputs, labels, num_classes=10):
        labels = torch.randint(0, num_classes, (np.size(labels, axis=0),))
        return inputs, labels
    
    @staticmethod
    def defendFlipAttack( global_model, client_models, plot_name="result", verbose=False):
        """
        Identifies malicious clients with flipping label attack
       
        """    
        label_sets = []
        num_classes = 1
        layer_name = "fc.weight"
        for source_class in range(num_classes):            
            param_diff = []
            global_params = list(global_model.state_dict()[layer_name])[source_class].cpu()
            for client in client_models:
                client_params = list(client.state_dict()[layer_name])[source_class].cpu()
                gradient = np.array([x for x in np.subtract(global_params, client_params)]).flatten()
                param_diff.append(gradient)

            scaler = StandardScaler()
            scaled_param_diff = scaler.fit_transform(param_diff)
            pca = PCA(2)
            dim_reduced_gradients = pca.fit_transform(scaled_param_diff)

            labels = fclusterdata(dim_reduced_gradients, t=2, criterion="maxclust")
            
            # Count the occurrences of each cluster label
            unique_labels, counts = np.unique(labels, return_counts=True)
            # Determine the most common cluster
            most_common_cluster = unique_labels[np.argmax(counts)]
            new_labels = np.where(labels == most_common_cluster, 1, 2)

            label_sets.append(new_labels)
        print("Label sets:", label_sets)
        malicious_clients = np.any(np.array(label_sets) - 1, axis=0) # maps most common label to 1, and other to 2
        print("Malicious clients:", malicious_clients)
        if plot_name:
            Flutils.plot_gradients_2d(dim_reduced_gradients, malicious_clients, plot_name)

        return [client_models[i] for i in range(len(client_models)) if malicious_clients[i] == False]
    
    @staticmethod
    def plot_gradients_2d(gradients, labels, name="fig.png"):
        for i in range(len(gradients)):
            gradient = gradients[i]
            params = (("blue", "x") if labels[i] else ("orange", "."))
            color, marker = params
            plt.scatter(gradient[0], gradient[1], color=color, marker=marker, s=50, linewidth=1)
        #plt.savefig(f"figures/{name}")
        # clear the plot
        plt.show()
        plt.clf()
