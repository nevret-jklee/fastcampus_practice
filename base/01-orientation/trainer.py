from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim


# Model을 학습하기 위한 코드
class Trainer():
    def __init__(self, model, optimizer, crit):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.crit = crit  # criterion = loss

    def _train(self, x, y, config):
        self.model.train()

        # Shuffle before begin
        # |x| = (batch_size, 784)
        '''
        random permutation: x.size(0)만큼의 무작위 수열을 만들어라 (즉, shuffling)
        shuffling되어 있는 indices대로 index를 select해서 batch size만큼 split한다.
        '''
        indices = torch.randperm(x.size(0), device=x.device)      
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)
        # print(len(x))
        # len(x), len(y) = 750 (if batch_size=64, 48000/64 = 750)
        # batch size만큼 split해서 tuple로 x에 저장한다.

        total_loss = 0

        # iteration for문
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())   # .squeeze(): (bs, 1) -> (bs, )

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:   # verbose: 얼마나 수다스러울것인가
                print("Train Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)   
            '''
            float를 안씌운 loss_i는 tensor가 돼서 total_loss도 tensor가 되고 이는 computation graph가 물려있다는 의미이므로
            엄청난 메모리를 잡아먹게되어 memory leak이 일어난다. (tensor의 정보량을 지우기 위해 float를 씌워줌)
            '''
        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Trun on the no_grad mode to make more efficiently. (gradient가 필요없으므로)
        with torch.no_grad():
            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())
                '''
                validation에는 zero_grad() -> backward() -> step()이 필요없음.
                학습이 아니기때문에 단순히 total loss만 구하면 됨
                '''
                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))

                total_loss += float(loss_i) # valid에는 float()가 의미없지만 똑같이 맞춰주기 위해

            return total_loss / len(x)


    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        # |train_data| = [(bs, 784), (bs, 1)]
        # |valid_data|
        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e   valid_loss=%.4e   lowest_loss=%.4e" %(
                epoch_index+1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore th best model.
        self.model.load_state_dict(best_model)


