import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # TODO: Complete this function
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_init = nn.Linear(embed_size, hidden_size)  # Project features to hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        batch_size = features.size(0)
        h0 = self.hidden_init(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)  # Cell
        hiddens, _ = self.lstm(inputs, (h0, c0))
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        predicted_sentence = []
        temperature = 0.7
        # Init states with inputs (features)
        batch_size = inputs.size(0)
        features = inputs.squeeze(1)  # [batch_size, embed_size]
        h0 = self.hidden_init(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(inputs.device)
        states = (h0, c0)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            probs = F.softmax(outputs / temperature, dim=1)
            predicted = torch.multinomial(probs, 1)
            pred_id = predicted.item()
            predicted_sentence.append(pred_id)
            if pred_id == 1:  # Stop at <end>
                break
            inputs = self.embed(predicted.squeeze(1)).unsqueeze(1)
        return predicted_sentence