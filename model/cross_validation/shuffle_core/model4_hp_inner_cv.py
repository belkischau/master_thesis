import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import Dataset, DataLoader, random_split,SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, f1_score
import sys
import os


# Get cpu, gpu or mps device for training.
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



#================================
#|	       		      	|
#|          FUNCTIONS           |
#|	       		       	|
#================================



def one_hot_encode_sequence(sequence):
    aa_to_int = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    encoded_sequence = torch.zeros(len(sequence), len(aa_to_int))
    for i, aa in enumerate(sequence):
        if aa == "X":
            encoded_sequence[i][:] = 1
        elif aa == "B":
            encoded_sequence[i][aa_to_int['D']] = 1
            encoded_sequence[i][aa_to_int['N']] = 1
        elif aa == "Z":
            encoded_sequence[i][aa_to_int['E']] = 1
            encoded_sequence[i][aa_to_int['Q']] = 1
        elif aa == "J":
            encoded_sequence[i][aa_to_int['I']] = 1
            encoded_sequence[i][aa_to_int['L']] = 1
        elif aa == "U":
            encoded_sequence[i][aa_to_int['C']] = 1
        else: 
            encoded_sequence[i][aa_to_int[aa]] = 1
    return encoded_sequence


class PeptideDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.peptides = self.dataframe['extended'].values
        self.targets = self.dataframe['target'].values
        self.encodeds = self.dataframe['encoding'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]
        target = self.targets[idx]
        encoding = self.encodeds[idx]

        # One-hot encode the peptide sequence
        return encoding, target 


class PeptideClassifier(torch.nn.Module):
    def __init__(self, h1=68, h2=12, dropout=0.3):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20 * in_features, h1)
        self.dropout = torch.nn.Dropout(p=dropout) #2) Add dropout
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

        
    def forward(self, x):

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.dropout(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = F.dropout(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x


class EarlyStopping:
    def __init__(self, patience = 7, path = 'checkpoint/checkpoint.pt', min_delta = 0.1) -> None:
        self.counter = 0
        self.patience = patience 
        self.early_stop = False
        self.checkpoint_file = path
        self.min_delta = min_delta
        self.path = path

    # evoke the function 
    def __call__(self, train_loss, valid_loss):
        loss_diff = (valid_loss - train_loss) / train_loss 
        if loss_diff > self.min_delta: 
            self.counter += 1
            if self.counter >= self.patience: 
#                torch.save(model.state_dict(), self.path)
                self.early_stop = True


def train(train_subset, valid_subset, h1, h2, lr, batch_size, dropout, checkpoint_path):
    model = PeptideClassifier(int(h1), int(h2), float(dropout))
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model)


    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = float(lr))

#    test_abs = int(len(trainset) * 0.8)
#    train_subset, val_subset = train_test_split(data_df, test_size=0.2, random_state = 42)

    train_loader = DataLoader(train_subset, batch_size=int(batch_size), shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=int(batch_size), shuffle=True)

    train_losses = []
    train_accuracies = []
    valid_losses = [] 
    valid_accuracies = []

    print("starting training from the function")
    for epoch in range(100):
        train_loss = 0.0
        epoch_steps = 0

        y_train_list = []
        train_predicted_list = []
        train_correct = 0

        for i, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train_list.extend(y_train.detach().cpu().numpy())

            # Zero the gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            train_output = model(x_train)
            train_output = train_output.reshape([len(x_train), -1])
            loss = criterion(train_output, y_train.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_train.size(0)
            threshold = torch.tensor([0.5]).to(device)

            train_predicted = (train_output>threshold).int()*1

            train_predicted_squeeze = torch.squeeze(train_predicted)
            train_predicted_squeeze = train_predicted_squeeze.type(torch.int64)
            train_predicted_list.extend(train_predicted_squeeze.detach().cpu().numpy())

            train_correct += (train_predicted_squeeze.detach().cpu().numpy() == y_train.detach().cpu().numpy()).sum()

#            if i % 2000 == 1999:  # print every 2000 mini-batches
#                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
#                                                train_loss / epoch_steps))
#                train_loss = 0.0

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = train_correct / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        val_steps = 0

        y_valid_list = []
        valid_predicted_list = []

        for i, (x_valid, y_valid) in enumerate(valid_loader):
            with torch.no_grad():
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                y_valid_list.extend(y_valid.detach().cpu().numpy())

                valid_output = model(x_valid)
                valid_output = valid_output.reshape([len(x_valid), -1])

                loss = criterion(valid_output, y_valid.float().view(-1, 1))
                valid_loss += loss.item() * x_valid.size(0)

                valid_predicted = (valid_output>threshold).int()*1
                valid_predicted_squeeze = torch.squeeze(valid_predicted)
                valid_predicted_squeeze = valid_predicted_squeeze.type(torch.int64)
                valid_predicted_list.extend(valid_predicted_squeeze.detach().cpu().numpy())

                valid_correct += (valid_predicted_squeeze.detach().cpu().numpy() == y_valid.detach().cpu().numpy()).sum()


        valid_loss /= len(valid_loader.sampler)
        valid_losses.append(valid_loss)
        valid_accuracy = valid_correct / len(valid_loader.sampler)
        valid_accuracies.append(valid_accuracy)


        if ((epoch +1) % 5 == 0) or (epoch == 0):
           print("Epoch %2i : Train Loss %f , Valid Loss %f , Train Acc %f , Valid Acc %f" % (
                epoch+1, train_losses[-1], valid_losses[-1], train_accuracies[-1] ,valid_accuracies[-1]))

        early_stopping(train_losses[-1], valid_losses[-1])

        if early_stopping.early_stop:         
            torch.save(model.state_dict(), f'{checkpoint_path}')
            print(f"Early stop at epoch: {epoch+1}. ")
            print(f"Final training statistics: \nTrain Loss %f | Valid Loss %f | Train Acc %f | Valid Acc %f" % (
                train_losses[-1], valid_losses[-1], train_accuracies[-1] ,valid_accuracies[-1]))
            break

    print("Finished 1 inner loop training")

    return h1, h2, lr, batch_size, dropout, valid_accuracy



def test_best_model(checkpoint, best_result):
    best_trained_model = PeptideClassifier(best_result["l1"], best_result["l2"], best_result["dropout"])
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)

#    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    checkpoint_path = checkpoint

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    testloader = DataLoader(testset, batch_size=best_result["batch_size"], shuffle=False)

    correct = 0
    total = 0
    y_test_list = []
    test_predicted_list = []
    y_proba = []
    test_prob = []

    with torch.no_grad():
        for (x_test, y_test) in testloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            test_output = best_trained_model(x_test)
            
            test_predicted = (batch_mean_prob>threshold).int()*1
            test_predicted_squeeze = torch.squeeze(test_predicted)
            test_predicted_squeeze = test_predicted_squeeze.type(torch.int64)
            test_correct += (test_predicted_squeeze.detach().cpu().numpy() == y_test.detach().cpu().numpy()).sum()

    test_prob_out = np.array([float(test_prob[i]) for i in range(len(test_prob))])
    y_test_out = np.array([float(y_test_list[i]) for i in range(len(y_test_list))])

    test_accuracy = test_correct / len(testloader.dataset)
    print("Best trial test set accuracy: {}".format(test_accuracy))

    auc = roc_auc_score(y_test_out, test_prob_out)
    print(f"AUC score of the trial test set: {auc}")

    return auc

def inner_loop(num_samples=10, max_num_epochs=100, gpus_per_trial=2):
    config = {
        "h1": tune.choice([16, 32, 64, 128]),
        "h2": tune.choice([4, 8, 16, 32]),
        "lr": tune.choice([0.1, 0.001, 0.0001, 0.0001]),
        "batch_size": tune.choice([8, 16, 32]),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

#====================== Main ===========================================


parser = argparse.ArgumentParser(description="Used for fine-tune hyperparameter")

parser.add_argument("-train", "--training", action="store", type=str, help="training data for hyperparameter tuning in the inner loop. (3 partition)")
parser.add_argument("-valid", "--validation", action="store", type=str, help="validation data which keep the same in the inner loop")
parser.add_argument("-test", "--testing", action="store", type=str, help="testing data which is the outer loop. ")
parser.add_argument("-lr", "--learning_rate", action="store", type=float, default=0.0001, help="learning rate: 0.01, 0.001, 0.0001, 0.00001")
#parser.add_argument("-o", "--optimizer", action="store", type=str, default='Adam', help="optimizer: Adam or SGD (Stochastic Gradient Descent)")
parser.add_argument("-d", "--dropout", action="store", type=float, default=0.3, help="dropuout rate")
parser.add_argument("-h1", "--hidden_feature1", action="store", type=int, default=44, help="number of hidden neurons in the first layer")
parser.add_argument("-h2", "--hidden_feature2", action="store", type=int, default=12, help="number of hidden neurons in the second layer")
parser.add_argument("-b", "--batch_size", action="store", type=int, default=32, help="batch_size")


args = parser.parse_args()
print(args)

train_file = args.training
valid_file = args.validation
test_file = args.testing 

lr = float(args.learning_rate)
dropout = float(args.dropout)
h1 = int(args.hidden_feature1)
h2 = int(args.hidden_feature2)
batch_size = int(args.batch_size)


data_path = "../../../dataset/model/partition"

'''

f1 = pd.read_csv(f"{data_path}/{train_file.split(', ')[0]}_9mer.csv")
f2 = pd.read_csv(f"{data_path}/{train_file.split(', ')[1]}_9mer.csv")
f3 = pd.read_csv(f"{data_path}/{train_file.split(', ')[2]}_9mer.csv")

trainset_df = pd.concat([f1, f2, f3])
subtest_df = pd.read_csv(f"{data_path}/{subtest_file}_9mer.csv")

subtrainset = PeptideDataset(trainset_df)
subtestset = PeptideDataset(subtest_df)
'''

in_features = 21

print("========== start cross validation  =============")


data_path = "../../../dataset/model/partition"
files = ['c000_9mer_shuffle.csv', 'c001_9mer_shuffle.csv', 'c002_9mer_shuffle.csv', 'c003_9mer_shuffle.csv', 'c004_9mer_shuffle.csv']


#h1 = [16, 32, 64, 128]
#h2 = [4, 8, 16, 32]
#lr = [0.01, 0.001, 0.0001, 0.00001]
#batch_size = [8, 16, 32, 64]
#dropout = [0.1, 0.2, 0.3]


all_file = ['c000', 'c001', 'c002', 'c003', 'c004']
#all_file.remove(test_file)
#all_file.remove(subtest_file)


checkpoint_path = f"outer{test_file[-1]}/checkpoint"
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

early_stopping = EarlyStopping(patience = 7, path = f"{checkpoint_path}/checkpoint.pt")

big_dict = list()

#for f in range(4):
#    sub_train = all_file.copy()
#    sub_train.remove(train_file.split(', ')[f])

f1 = pd.read_csv(f"{data_path}/{train_file.split(', ')[0]}_9mer_shuffle.csv")
f2 = pd.read_csv(f"{data_path}/{train_file.split(', ')[1]}_9mer_shuffle.csv")
f3 = pd.read_csv(f"{data_path}/{train_file.split(', ')[2]}_9mer_shuffle.csv")

train_df = pd.concat([f1, f2, f3])
train_df['encoding'] = train_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
trainset = PeptideDataset(train_df)

#    valid = train_file.split(', ')[f]
#    print(f"sub train file = {sub_train}, sub valid file = {sub_valid}")

valid_df = pd.read_csv(f"{data_path}/{valid_file}_9mer_shuffle.csv")
valid_df['encoding'] = valid_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
validset = PeptideDataset(valid_df)

#    print(f"Subtest test file = {subtest_file}")

train_h1, train_h2, train_lr, train_batch_size, train_dropout, train_valid_accuracy = train(trainset, validset, h1, h2, lr, batch_size, dropout, f"{checkpoint_path}/{test_file}_{valid_file}_{h1}_{h2}_{lr}_{batch_size}_{dropout}.pt")
print("Finish training")
print(train_h1, train_h2, train_lr, train_batch_size, train_dropout, train_valid_accuracy)
    
'''

    number = 0
    hp_dict = {"h1":[], "h2":[], "lr":[], "batch_size":[], "dropout":[], "valid_acc":[]}

    for a in h1:
        for b in h2:
            for c in lr:
                for d in batch_size:
                    for e in dropout:
                        print("start trainin!!!!!!")
                        train_h1, train_h2, train_lr, train_batch_size, train_dropout, train_valid_accuracy = train(sub_trainset, sub_validset, a, b, c, d, e, f"checkpoint/{number}.pt")
                        print("Final statistics:",  train_h1, train_h2, train_lr, train_batch_size, train_dropout, train_valid_accuracy)
#                        hp_dict["h1"].extend(int(train_h1))
#                        hp_dict["h2"].extend(train_h2)
#                        hp_dict["lr"].extend(train_lr)
#                        hp_dict["batch_size"].extend(batch_size)
#                        hp_dict["dropout"].extend(train_dropout)
#                        hp_dict["valid_acc"].extend(train_valid_accuracy)
                        number += 1


#                        break 
#                    break
#                break
#            break
#        break
    big_dict.append(hp_dict)
#    break





'''


















'''
for i, test in enumerate(files):
    print(f"\n\nOuter fold {i+1}/5")

    test_df = pd.read_csv(f"{data_path}/c00{i}_9mer.csv")
    test_df['encoding'] = test_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
    test_dataset = PeptideDataset(test_df)
#    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    file_list = ['c000_9mer.csv', 'c001_9mer.csv', 'c002_9mer.csv', 'c003_9mer.csv', 'c004_9mer.csv']
    file_list.remove(f"c00{i}_9mer.csv")
    print(f"Files in inner fold = {file_list}")

    for n, f in enumerate(file_list):
        train_valid_list = file_list.copy()
        print(f"\nOuter fold {i+1}/5, inner fold {n+1}/4")
        print(f"Valid file = {data_path}/{train_valid_list[n]}")
        train_file_list = train_valid_list.copy()
        train_file_list.remove(f"{train_valid_list[n]}")
        print(f"Train files = {train_file_list}")

        train0 = pd.read_csv(f"{data_path}/{train_file_list[0]}")
        train1 = pd.read_csv(f"{data_path}/{train_file_list[1]}")
        train2 = pd.read_csv(f"{data_path}/{train_file_list[2]}")

        train_df = pd.concat([train0, train1, train2])
        valid_df = pd.read_csv(f"{data_path}/{train_valid_list[n]}")

        train_df['encoding'] = train_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
        valid_df['encoding'] = valid_df['extended'].apply(lambda x: one_hot_encode_sequence(x))

        trainset = PeptideDataset(train_df)
        testset = PeptideDataset(valid_df)

        inner_loop(num_samples=2, max_num_epochs=100, gpus_per_trial=0)

        break 
    break


'''












'''



def inner_train():
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        train_correct = 0
        train_accuracies_batches = []
        total = 0
        y_train_list = []
        train_predicted_list = []
        
        for a, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train_list.extend(y_train.detach().cpu().numpy())

            optimizer.zero_grad()

            train_output = model(x_train)
            train_output = train_output.reshape([len(x_train), -1])

            loss = criterion(train_output, y_train.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_train.size(0)
            threshold = torch.tensor([0.5]).to(device)
            threshold = torch.tensor([threshold]).to(device)

            train_predicted = (train_output>threshold).int()*1

            train_predicted_squeeze = torch.squeeze(train_predicted)
            train_predicted_squeeze = train_predicted_squeeze.type(torch.int64)
            train_predicted_list.extend(train_predicted_squeeze.detach().cpu().numpy())

            train_correct += (train_predicted_squeeze.detach().cpu().numpy() == y_train.detach().cpu().numpy()).sum()



'''
