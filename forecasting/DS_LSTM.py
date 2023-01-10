from torch import zeros
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.autograd import Variable


class DS_LSTM(Module):

    def __init__(self, input_size, hidden_size, learning_rate, num_layers=1, num_classes=1):
        super(DS_LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = Linear(hidden_size, self.num_classes)
        self.criterion = MSELoss()    # mean-squared error for regression
        self.optimizer = Adam(self.parameters(), lr=learning_rate)


    def forward(self, x):
        h_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

    def fit(self, trainX, trainY):
        # Train the model
        outputs = self(trainX)
        self.optimizer.zero_grad()
        # obtain the loss function
        loss = self.criterion(outputs, trainY)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, data):
        return self(data).detach().numpy()