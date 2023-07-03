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
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, f1_score
import sys
import os
#from ray import tune
#from ray.air import Checkpoint, session
#from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import matthews_corrcoef


# Get cpu, gpu or mps device for training.
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



#================================
#|	       		      	|
#|          FUNCTIONS           |
#|	       		       	|
#================================

# Set the random seed for NumPy
np.random.seed(42)

# Set the random seed for PyTorch
torch.manual_seed(24)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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

#    print("starting training from the function")
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
            best_model_path = f'{checkpoint_path}'
            print(f"Early stop at epoch: {epoch+1}. ")
            print(f"Final training statistics: \nTrain Loss %f | Valid Loss %f | Train Acc %f | Valid Acc %f" % (
                train_losses[-1], valid_losses[-1], train_accuracies[-1] ,valid_accuracies[-1]))
            early_stopping.early_stop = False
            break

        if (epoch +1) == 100:
            torch.save(model.state_dict(), f'{checkpoint_path}')
            best_model_path = f'{checkpoint_path}'


#    print("Finished 1 inner loop training")

    return h1, h2, lr, batch_size, dropout, valid_accuracy, best_model_path



def test_best_model(testset, h1, h2, batch_size, dropout, checkpoint):
    best_trained_model = PeptideClassifier(h1, h2, dropout)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)

#    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    checkpoint_path = checkpoint

#    model_state, optimizer_state = torch.load(checkpoint_path)
#    best_trained_model.load_state_dict(model_state)
    best_trained_model.load_state_dict(torch.load(checkpoint))
    best_trained_model.eval()

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    test_correct = 0
    total = 0
    x_test_list = []
    y_test_list = []
    test_predicted_list = []
    y_proba = []
    test_prob = []

    with torch.no_grad():
        for (x_test, y_test) in testloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            x_test_list.extend(x_test.detach().cpu().numpy())
            y_test_list.extend(y_test.detach().cpu().numpy())

            test_output = best_trained_model(x_test)
            test_output = test_output.reshape([len(x_test), -1])
            test_prob.extend(test_output)
            
            test_predicted = (test_output>0.5).int()*1
            test_predicted_squeeze = torch.squeeze(test_predicted)
            test_predicted_squeeze = test_predicted_squeeze.type(torch.int64)
            test_predicted_list.extend(test_predicted_squeeze.detach().cpu().numpy())
            test_correct += (test_predicted_squeeze.detach().cpu().numpy() == y_test.detach().cpu().numpy()).sum()

    test_prob_out = np.array([float(test_prob[i]) for i in range(len(test_prob))])
    y_test_out = np.array([float(y_test_list[i]) for i in range(len(y_test_list))])

    test_accuracy = test_correct / len(testloader.dataset)
    print("Best trial test set accuracy: {}".format(test_accuracy))

    auc = roc_auc_score(y_test_out, test_prob_out)
    print(f"AUC score of the trial test set: {auc}")

    confusion_matrix = metrics.confusion_matrix(y_test_list, test_predicted_list)    
    print(f"Confusion matrix: {confusion_matrix}\n")
    print("tn, fp, fn, tp")
    print(metrics.confusion_matrix(y_test_list, test_predicted_list).ravel())
    print('\n')
    mcc = matthews_corrcoef(y_test_list, test_predicted_list)
    print(f"MCC = {mcc}\n")
    test_length = len(testloader.dataset)
    return test_accuracy, auc, mcc, x_test_list, y_test_list, test_prob_out, test_length



def prepare_train_validset(train_files, valid_file):
    f1 = pd.read_csv(f"{data_path}/{train_files[0]}_9mer.csv")
    f2 = pd.read_csv(f"{data_path}/{train_files[1]}_9mer.csv")
    f3 = pd.read_csv(f"{data_path}/{train_files[2]}_9mer.csv")


    train_df = pd.concat([f1, f2, f3])
    train_df['encoding'] = train_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
    trainset = PeptideDataset(train_df)

    valid_df = pd.read_csv(f"{data_path}/{valid_file}_9mer.csv")
    valid_df['encoding'] = valid_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
    validset = PeptideDataset(valid_df)

    return trainset, validset


def decode_sequence(encoded_sequence):
    int_to_aa = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y'}
    decoded_sequence = ""
    for encoded_aa in encoded_sequence:
        aa_index = torch.argmax(encoded_aa).item()
        decoded_aa = int_to_aa[aa_index]
        decoded_sequence += decoded_aa
    return decoded_sequence

def extract_result(x_list, true_list, predict_list):

    '''
    To ectract the peptide which are TP ,TN...
    Inputs 
    x_list : list()
        true peptide list
    true_list : list()
        true label 
    predict_list : list()
        predicted label 

    Output: 
    tp : list()
        true positive peptide list 
    tn : list()
        true negative peptide list 
    fp : list()
        false positive peptide list 
    fn : list()
        false negative peptide list 
    positive : list()
        a list of peptide which the model classifies them as positive (TP + FP)
    negative : list()
        a list of peptide which the model classifies them as negative (FN + TN)
    ''' 

#    positive = []
    tp = []
    fp = []
#    negative = []
    tn = []
    fn = []

    for i, x in enumerate(x_list): 
        sequence = decode_sequence(x)
        if predict_list[i] == true_list[i]:
#            x_correct.append(sequence) 
            if true_list[i] == 1:
                tp.append(sequence) 
            else:
                tn.append(sequence)
        else:
#            x_incorrect.append(sequence)
            if true_list[i] == 0:
                fp.append(sequence)
            else:
                fn.append(sequence)
    return tp, tn, fp, fn


def write_peptide(peptide_list, file):
    with open(file, 'w') as f:
        for p in peptide_list: 
            f.write(str(p)) 
            f.write('\n')

def plot_ROCAUC_curve(y_truth, y_proba, fig_size, out_file):
    
    '''
    Plots the Receiver Operating Characteristic Curve (ROC) and displays Area Under the Curve (AUC) score.
    
    Args:
        y_truth: ground truth for testing data output
        y_proba: class probabilties predicted from model
        fig_size: size of the output pyplot figure
    
    Returns: outfile with plot
    '''
    
    fpr, tpr, threshold = roc_curve(y_truth, y_proba)
    auc_score = roc_auc_score(y_truth, y_proba)
    txt_box = "AUC Score: " + str(round(auc_score, 4))
    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1],'--')
    plt.annotate(txt_box, xy=(0.65, 0.05), xycoords='axes fraction')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.savefig(out_file)








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

in_features = 21

parameter0 = [[128, 32, 0.0001, 32, 0.1], 
              [128, 16, 0.0001, 32, 0.2], 
              [128, 16, 0.0001, 32, 0.3], 
              [128, 32, 0.0001, 32, 0.3]]


parameter1 = [[128, 16, 0.0001, 32, 0.2],
              [128, 32, 0.0001, 32, 0.1], 
              [128, 16, 0.0001, 32, 0.3], 
              [128, 32, 0.0001, 32, 0.3]]


parameter2 = [[128, 32, 0.0001, 32, 0.3],
              [128, 16, 0.0001, 32, 0.1], 
              [128, 32, 0.0001, 32, 0.1], 
              [128, 32, 0.0001, 32, 0.1]]


parameter3 = [[128, 16, 0.0001, 32, 0.3],
              [128, 32, 0.00001, 32, 0.3], 
              [128, 16, 0.0001, 32, 0.1], 
              [128, 32, 0.0001, 32, 0.3]]

parameter4 = [[128, 32, 0.0001, 32, 0.3], 
              [128, 32, 0.0001, 32, 0.3], 
              [128, 32, 0.0001, 32, 0.2], 
              [128, 32, 0.0001, 32, 0.3]]


parameter_list = [parameter0, parameter1, parameter2, parameter3, parameter4]

print("========== start cross validation  =============")


data_path = "../../../dataset/model/partition"
files = ['c000_9mer_shuffle.csv', 'c001_9mer_shuffle.csv', 'c002_9mer_shuffle.csv', 'c003_9mer_shuffle.csv', 'c004_9mer_shuffle.csv']


#h1 = [16, 32, 64, 128]
#h2 = [4, 8, 16, 32]
#lr = [0.01, 0.001, 0.0001, 0.00001]
#batch_size = [8, 16, 32, 64]
#dropout = [0.1, 0.2, 0.3]


checkpoint_path = f"outer_checkpoint_32"
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

plot_path = "plot_outer_32"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

peptide_out_path = "peptide_out_32"
if not os.path.exists(peptide_out_path):
    os.mkdir(peptide_out_path)



early_stopping = EarlyStopping(patience = 7, path = f"{checkpoint_path}/checkpoint.pt")

all_file = ['c000', 'c001', 'c002', 'c003', 'c004']


all_test_correct = 0
all_y_test_list = []
all_test_predicted_list = []

test_prob_out_list = []
y_test_out_list = []

all_tp = []
all_tn = []
all_fp = []
all_fn = []

for t in range(len(all_file)): 
    test = all_file[t]
    test_df = pd.read_csv(f"{data_path}/{test}_9mer_shuffle.csv")
    test_df['encoding'] = test_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
    testset = PeptideDataset(test_df)

    train_files = all_file.copy()
    train_files.remove(test)
    print(f"test file = {test}")
#    print(f"inner files = {train_files}")


    best_h1_list = []
    best_h2_list = []
    best_lr_list = []
    best_batch_size_list = []
    best_dropout_list = []
    best_model_path_list = []
    test_predicted_list = []
    test_prob_list = []


    for v in range(len(train_files)): 
        valid_file = train_files[v]
        inner_files = train_files.copy()
        inner_files.remove(valid_file)
        print(f"valid file = {valid_file}, train files = {inner_files}")
        trainset, validset = prepare_train_validset(inner_files, valid_file)

        checkpoint_name = f"c00{t}_c{v}.pt"
        best_h1 = parameter_list[t][v][0]
        best_h2 = parameter_list[t][v][1]
        best_lr = parameter_list[t][v][2]
        best_batch_size = parameter_list[t][v][3]
        best_dropout = parameter_list[t][v][4]
#        print(best_h1, best_h2, best_lr, best_batch_size, best_dropout)

        print(f"=============== Start training the models {v+1}/4 ========================")

        best_h1, best_h2, best_lr, best_batch_size, best_dropout, valid_accuracy, best_model_path =  train(trainset, 
                          validset, best_h1, best_h2, best_lr, best_batch_size, best_dropout,
                          f"{checkpoint_path}/{checkpoint_name}")
        best_h1_list.append(best_h1)
        best_h2_list.append(best_h2)
        best_lr_list.append(best_lr)
        best_dropout_list.append(best_dropout)
        best_model_path_list.append(best_model_path)

#    best_model_path_list = ['outer_checkpoint/c000_c0.pt', 'outer_checkpoint/c000_c1.pt', 'outer_checkpoint/c000_c2.pt', 'outer_checkpoint/c000_c3.pt', 'outer_checkpoint/c000_c4.pt']
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    y_test_list = []
    test_pred_prob = []
    test_predicted_list = []
    test_correct = 0
    prob_mean_list = []
    x_test_list = []


    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        x_test_list.extend(x_test)
        y_test_list.extend(y_test.detach().cpu().numpy())
        test_pred_prob = torch.empty((len(x_test)), dtype=torch.float64)        

        for m in range(len(train_files)):
            model = PeptideClassifier(best_h1_list[m], best_h2_list[m], best_dropout_list[m])
            model.to(device)
            model.load_state_dict(torch.load(f"{best_model_path_list[m]}"))
#            model.load_state_dict(torch.load(f"outer_checkpoint_32/c00{t}_c{m}.pt"))
            model.eval()

            test_output = model(x_test)
            test_output = test_output.reshape([len(x_test), -1])

            if m == 0: 
                model_0 = torch.flatten(test_output)
            elif m == 1:
                model_1 = torch.flatten(test_output)
            elif m == 2:
                model_2 = torch.flatten(test_output)
#                test_pred_prob = torch.cat((model_0, model_1, model_2)).reshape((m+1), len(x_test))
            elif m == 3:
                model_3 = torch.flatten(test_output)
                test_pred_prob = torch.cat((model_0, model_1, model_2, model_3)).reshape((m+1), len(x_test))
#        print(model_0)
#       print(model_1)
#        print(model_2)
#       print(model_3)
#        print(test_pred_prob)

        batch_mean_prob = test_pred_prob.mean(dim=0)
        batch_mean_prob = batch_mean_prob.reshape([len(x_test), -1]) 
        prob_mean_list.extend(batch_mean_prob)

        test_predicted = (batch_mean_prob>0.5).int()*1
        test_predicted_squeeze = torch.squeeze(test_predicted)
        test_predicted_squeeze = test_predicted_squeeze.type(torch.int64)
        test_predicted_list.extend(test_predicted_squeeze.detach().cpu().numpy())

        test_correct += (test_predicted_squeeze.detach().cpu().numpy() == y_test.detach().cpu().numpy()).sum()

    test_accuracy = test_correct / len(test_loader.dataset)
    print(f"\nTest accuracy: {test_accuracy}")
    print(f"Testing f1 score: {f1_score(y_test_list, test_predicted_list)}")

    confusion_matrix = metrics.confusion_matrix(y_test_list, test_predicted_list)
    print(f"Confusion matrix: {confusion_matrix}\n")
    print("tn, fp, fn, tp")
    print(metrics.confusion_matrix(y_test_list, test_predicted_list).ravel())
    print('\n')
    print(classification_report(y_test_list, test_predicted_list))

    test_prob_out = np.array([float(prob_mean_list[y]) for y in range(len(prob_mean_list))])
    y_test_out = np.array([float(y_test_list[z]) for z in range(len(y_test_list))])
    print(f'AUC score: {roc_auc_score(y_test_out, test_prob_out)}')

    mcc = matthews_corrcoef(y_test_list, test_predicted_list)
    print(f"\nMCC = {mcc}\n")


    tp, tn, fp, fn = extract_result(x_test_list, y_test_list, test_predicted_list)
    write_peptide(tp, f"{peptide_out_path}/fold{t}_tp.out")
    write_peptide(tn, f"{peptide_out_path}/fold{t}_tn.out")
    write_peptide(fp, f"{peptide_out_path}/fold{t}_fp.out")
    write_peptide(fn, f"{peptide_out_path}/fold{t}_fn.out")

    plot_ROCAUC_curve(y_test_out, test_prob_out, (8, 8), f"{plot_path}/AUC_fold{t}.png")


    all_test_correct += test_correct
    all_y_test_list.extend(y_test_list)
    all_test_predicted_list.extend(test_predicted_list)

    test_prob_out_list.extend(test_prob_out)
    y_test_out_list.extend(y_test_out)

    all_tp.extend(tp)
    all_tn.extend(tn)
    all_fp.extend(fp)
    all_fn.extend(fn)
#    break




## Final CV performance 

print("\nFinal CV performance: \n")

# accuracy & F1 score 
final_accuracy = all_test_correct / len(all_y_test_list)
print(f"\nTest accuracy: {final_accuracy}")
print(f"Testing f1 score: {f1_score(all_y_test_list, all_test_predicted_list)}")


# confusion matrix 
final_cf = metrics.confusion_matrix(all_y_test_list, all_test_predicted_list)
print(f"Confusion matrix: {final_cf}\n")
print("tn, fp, fn, tp")
print(metrics.confusion_matrix(all_y_test_list, all_test_predicted_list).ravel())
print('\n')
print(classification_report(all_y_test_list, all_test_predicted_list))

# AUC
print(f'AUC score: {roc_auc_score(y_test_out_list, test_prob_out_list)}\n')
plot_ROCAUC_curve(y_test_out_list, test_prob_out_list, (8, 8), f"{plot_path}/final_AUC.png")

# MCC
mcc = matthews_corrcoef(all_y_test_list, all_test_predicted_list)
print(f"MCC = {mcc}\n")

# get tp, tn, fp, fn peptides 
write_peptide(all_tp, f"{peptide_out_path}/final_tp.out")
write_peptide(all_tn, f"{peptide_out_path}/final_tn.out")
write_peptide(all_fp, f"{peptide_out_path}/final_fp.out")
write_peptide(all_fn, f"{peptide_out_path}/final_fn.out")





print("Finished evaluating! ")


# ===============  END  =================













'''

        best_h1, best_h2, best_lr, best_batch_size, best_dropout, valid_accuracy, best_model_path =  train(trainset, 
                          validset, best_h1, best_h2, best_lr, best_batch_size, best_dropout, 
                          f"{checkpoint_path}/{checkpoint_name}")
        best_batch_size = 32

        best_model_path = f"{checkpoint_path}/c00{t}_c{v}.pt"
        test_accuracy, auc, mcc, x_test_list, y_test_out, test_prob, test_length = test_best_model(trainset,
                                                           best_h1, best_h2,
                                                           best_batch_size, best_dropout,
                                                           best_model_path)

        test_prob_list.append(test_prob)
        print(y_test_out[:20])
#        if v == 0:
#            test_prob = torch.tensor(test_prob)
#            model_0 = torch.flatten(test_prob)
#            model_0 = test_prob
#            test_pred_prob = model_0
#        elif m == 1:
#            model_1 = torch.flatten(test_prob)
#
#        elif m == 2:
#            model_2 = torch.flatten(test_prob)
#        elif m == 3:
#            model_3 = torch.flatten(test_prob)
#            test_pred_prob = torch.cat((model_0, model_1, model_2, model_3)).reshape((m+1), test_length)

#        test_pred_prob = np.concatenate((model_0, model_1, model_2, model_3))

        print("Results:\n")
        print(test_accuracy, auc, mcc)

#        break

    print(test_prob_list[0][:10])
    print(test_prob_list[1][:10])
    print(test_prob_list[2][:10])
    print(test_prob_list[3][:10])

    test_pred_prob = np.concatenate((test_prob_list[0], test_prob_list[1], test_prob_list[2]), axis=0)
    print(test_pred_prob[:10])

#    mean_prob = test_pred_prob.mean(dim=0)
#    mean_prob = mean_prob.reshape([test_length, -1])
    mean_prob = np.mean(test_pred_prob, axis = 0)
    print(f"mean probability = {mean_prob[:10]}")


#    test_predicted = (mean_prob>0.5).int()*1
#    test_predicted_squeeze = torch.squeeze(test_predicted)
#    test_predicted_squeeze = test_predicted_squeeze.type(torch.int64)
#    test_predicted_list.append(test_predicted_squeeze.detach().cpu().numpy())

    test_predicted_list = np.round_(mean_prob)
    print(f"test predicted list = {test_predicted_list[:10]}")
#    test_correct += (test_predicted_squeeze.detach().cpu().numpy() == y_test.detach().cpu().numpy()).sum()
    test_correct = (test_predicted_list == y_test_out).sum()

    test_accuracy = test_correct / (test_length)
    print(f"\nTest accuracy: {test_accuracy}")
    print(f"Testing f1 score: {f1_score(y_test_out, test_predicted_list)}")

    confusion_matrix = metrics.confusion_matrix(y_test_out, test_predicted_list)
    print(f"Confusion matrix: {confusion_matrix}\n")
    print("tn, fp, fn, tp")
    print(metrics.confusion_matrix(y_test_out, test_predicted_list).ravel())
    print('\n')
    print(classification_report(y_test_out, test_predicted_list))

    test_prob_out = np.array([float(prob_mean_list[y]) for y in range(len(prob_mean_list))])
    y_test_out = np.array([float(y_test_out[z]) for z in range(len(y_test_out))])
    print(f'AUC score: {roc_auc_score(y_test_out, test_prob_out)}')

    tp, tn, fp, fn = extract_result(x_test_list, y_test_list, test_predicted_list)
    write_peptide(tp, f"{peptide_out_path}/fold{i}_tp.out")
    write_peptide(tn, f"{peptide_out_path}/fold{i}_tn.out")
    write_peptide(fp, f"{peptide_out_path}/fold{i}_fp.out")
    write_peptide(fn, f"{peptide_out_path}/fold{i}_fn.out")

    plot_ROCAUC_curve(y_test_out, test_prob_out, (16, 16), f"{plot_path}/AUC_fold{t}.png")

    break






'''







'''

        best_h1_list.append(best_h1)
        best_h2_list.append(best_h2)
        best_lr_list.append(best_lr)
        best_batch_size_list.append(best_batch_size)
        best_dropout_list.append(best_dropout)
        best_model_path_list.append(best_model_path)

        break

    print(f"best_model_path_list: = {best_model_path_list}")

    test_pred_prob = torch.empty((len(testset)), dtype=torch.float64)

    for m in range(len(best_model_path_list)):
        print(f"============ start testing the models {m+1}/4 ===================")
        test_accuracy, auc, y_test_list, test_prob = test_best_model(trainset, 
                                                           best_h1_list[m], best_h2_list[m],  
                                                           best_batch_size_list[m], best_dropout_list[m], 
                                                           best_model_path_list[m])
        all_acc_list.append(test_accuracy)
        all_auc.append(auc)
        all_y_test_list.extend(y_test_list)
        if m == 0: 
            model_0 = torch.flatten(test_prob)
        elif m == 1:
            model_1 = torch.flatten(test_prob)
        elif m == 2:
            model_2 = torch.flatten(test_prob)
        elif m == 3:
            model_3 = torch.flatten(test_prob)
            test_pred_prob = torch.cat((model_0, model_1, model_2, model_3)).reshape((m+1), len(x_test))

    if all_y_test_list[0] == all_y_test_list[1]:
        print("The order for testing is the same for each model. ")
    else: 
        print("The order for testing is NOT the same for each model. ")

    mean_prob = test_pred_prob.mean(dim=0)
    mean_prob = mean_prob.reshape([len(all_acc_list[0]), -1])

    test_predicted = (mean_prob>threshold).int()*1
    test_predicted_squeeze = torch.squeeze(test_predicted)
    test_predicted_squeeze = test_predicted_squeeze.type(torch.int64)
    test_predicted_list.extend(test_predicted_squeeze.detach().cpu().numpy())

    test_correct = (test_predicted_squeeze.detach().cpu().numpy() == y_test.detach().cpu().numpy()).sum()

    test_accuracy = test_correct / len(test_loader.dataset)
    print(f"\nTest accuracy: {test_accuracy}")
    print(f"Testing f1 score: {f1_score(all_y_test_list[0], test_predicted_list)}")
    confusion_matrix = metrics.confusion_matrix(all_y_test_list[0], test_predicted_list)
    print(f"Confusion matrix: {confusion_matrix}\n")
    print("tn, fp, fn, tp")
    print(metrics.confusion_matrix(all_y_test_list[0], test_predicted_list).ravel())
    print('\n')
    print(classification_report(y_test_list, test_predicted_list))

    test_prob_out = np.array([float(mean_prob[y]) for y in range(len(mean_prob))])
    y_test_out = np.array([float(y_test_list[0][z]) for z in range(len(y_test_list[0]))])
    print(f'AUC score: {roc_auc_score(y_test_out, test_prob_out)}')

    break

'''




'''


checkpoint_path = "checkpoint"
early_stopping = EarlyStopping(patience = 7, path = f"{checkpoint_path}/checkpoint.pt")

big_dict = list()

#for f in range(4):
#    sub_train = all_file.copy()
#    sub_train.remove(train_file.split(', ')[f])

f1 = pd.read_csv(f"{data_path}/{train_file.split(', ')[0]}_9mer.csv")
f2 = pd.read_csv(f"{data_path}/{train_file.split(', ')[1]}_9mer.csv")
f3 = pd.read_csv(f"{data_path}/{train_file.split(', ')[2]}_9mer.csv")

train_df = pd.concat([f1, f2, f3])
train_df['encoding'] = train_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
trainset = PeptideDataset(train_df)

#    valid = train_file.split(', ')[f]
#    print(f"sub train file = {sub_train}, sub valid file = {sub_valid}")

valid_df = pd.read_csv(f"{data_path}/{valid_file}_9mer.csv")
valid_df['encoding'] = valid_df['extended'].apply(lambda x: one_hot_encode_sequence(x))
validset = PeptideDataset(valid_df)

#    print(f"Subtest test file = {subtest_file}")

train_h1, train_h2, train_lr, train_batch_size, train_dropout, train_valid_accuracy = train(trainset, validset, h1, h2, lr, batch_size, dropout, f"checkpoint/{test_file}_{valid_file}_{h1}_{h2}_{lr}_{batch_size}_{dropout}.pt")
print("Finish training")
print(train_h1, train_h2, train_lr, train_batch_size, train_dropout, train_valid_accuracy)
   

'''








 
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