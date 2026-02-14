import torch 
from torch import nn
from torch.utils.data import TensorDataset
from src.processing.preprocessed import preprocess_data
DATA_PATH = "training_data.csv"
model_save_dir = "models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_normal_data = preprocess_data(DATA_PATH, model_save_dir)
dataset = TensorDataset(X_normal_data)
TRAIN_TENSOR = torch.FloatTensor(X_normal_data).to(DEVICE)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_neurons_layer, dropout_rate, num_layers):
        super(Autoencoder, self).__init__()
        #encoder
        encoder_layers =[]
        current_dim = input_dim
        next_dim = output_neurons_layer
        self.layer_sizes=[]
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            encoder_layers.append(nn.BatchNorm1d(next_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            self.layer_sizes.append(next_dim) # we are storing the layer sizes of encoder to use it in decoder(it does not have input_dim)
            current_dim = next_dim
            next_dim = max(5, next_dim//2)
        self.encoder = nn.Sequential(*encoder_layers)
        #decoder
        decoder_layer=[]
        reverse_sizes = self.layer_sizes[::-1] #we are reversing encoder to make the decoder
        current_dim = self.layer_sizes[-1]  #we will start with the last layer of encoder as the first layer of decoder      
        target_nn = reverse_sizes[1:] + [input_dim] #we will add input_dim at last so that we can reconstruct the input and we will remove the last layer of encoder as it is the output layer of encoder and input layer of decoder
        for target_dim in target_nn:
            decoder_layer.append(nn.Linear(current_dim, target_dim))
            if target_dim != num_layers-1: #we don't want to apply activation and dropout on the last layer of decoder
                decoder_layer.append(nn.BatchNorm1d(target_dim))
                decoder_layer.append(nn.ReLU())
                decoder_layer.append(nn.Dropout(dropout_rate))
            current_dim = target_dim
        self.decoder = nn.Sequential(*decoder_layer)
        self.apply(self._init_weights) # we are not going to use pretrained weights for the autoencoder, we will initialize the weights using kaiming uniform initialization which is suitable for ReLU activation function. This will help in faster convergence of the model.
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    def forward(self, x):
        return self.decoder(self.encoder(x))