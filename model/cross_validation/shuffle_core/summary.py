import os
import argparse


# Obtain the arguement from the command line
parser = argparse.ArgumentParser(description="Used for extract results from hyper-parameter tuning")

parser.add_argument("-d", "--directory", action="store", type=str, help="directory where the output files from hyperparameter tuning stored.")
parser.add_argument("-o", "--out_file", action="store", type=str, default="summary_ealry_stop.out", help="the output file. ")

args = parser.parse_args()

path = args.directory
file_list = os.listdir(path)
summary_file = args.out_file

with open(f"{summary_file}", 'w') as f:
    f.write("test_file, valid_file, epoch, h1, h2, lr, batch_size, dropuout, valid_acc\n")

#nf = 0

for file in file_list:
#    print(f"file readng = {file}")
    if os.path.isdir(f"{path}/{file}"):
        continue

    with open(f"{path}/{file}", 'r') as rf:
        header = str(file)[:-4]
        test = file.split("_")[0]
        valid = file.split("_")[1]
        

#    file = 'loss_0.01_Adam_0.1_32_12.out'
#    file = 'per_0.01_Adam_0.1_32_12.out'
#    with open('performance/output/early_stop/per_0.01_Adam_0.1_32_12.out') as rf:
#        if str(file).startswith("loss_"):
#            header = str(file)[5:-4]
#        elif str(file).startswith("per_"):
#            header = str(file)[4:-4]
#        flag = False
        with open(f"{summary_file}", 'a') as wf:
            for line in rf:
                if line.startswith("Early stop at "): 
                    epoch = int(line.split()[-1][:-1])
                if line.startswith("Epoch 100"):
                    epoch = int(100)
                if str(line)[0].isdigit():
                    h1 = int(line.split()[0])
                    h2 = int(line.split()[1])
                    lr = float(line.split()[2])
                    batch_size = int(line.split()[3])
                    dropout = float(line.split()[4])
                    valid_acc = float(line.split()[5])
            wf.write(f"{test}, {valid}, {epoch}, {h1}, {h2}, {lr}, {batch_size}, {dropout}, {valid_acc}\n")
#    nf += 1
#    if nf == 10: 
#        break


'''
#            lr = header.split("_")[0]
#            optimizer = header.split("_")[1]
            dropout = header.split("_")[2]
            try: 
                h1 = header.split("_")[3]
                h2 = header.split("_")[4]
            except IndexError as err:
                print(header)

            train_loss = 0
            flag = False
            for line in rf:
                if line.startswith("Train Loss"):
                    train_loss = float(line.split()[2])
                    valid_loss = float(line.split()[6])
                    train_acc = float(line.split()[10])
                    valid_acc = float(line.split()[14])
                    train_f1 = float(line.split()[18])
                    valid_f1 = float(line.split()[22])

#                    train_loss_list.append(train_loss)
#                    valid_loss_list.append(valid_loss)
#                    train_acc_list.append(train_acc)
#                    valid_acc_list.append(valid_acc)

                if line.startswith("Epoch 100"):
                    epoch = 100
                    train_loss = float(line.split()[5])
                    valid_loss = float(line.split()[9])
                    train_acc = float(line.split()[13])
                    valid_acc = float(line.split()[17])
                    train_f1 = float(line.split()[21][:-1])
                    valid_f1 = float(line.split()[24])

                if line.startswith("Test accuracy"):
                    test_acc = line.split()[2]
#                    test_acc_list.append(test_acc)

                if line.startswith("Early stop at"):
                    epoch = int(line.split()[-1][:-1])

                if flag: 
                    try: 
                        line = str(line)
                        tn = line.split()[0]
#                        fp = int(line.split()[1])
#                        fn = int(line.split()[2])
#                        tp = int(line.split()[-1][:-1])

                        if len(tn) < 2:
                            tn = int(line.split()[1])
                            fp = int(line.split()[2])
                            fn = int(line.split()[3])
                            tp = int(line.split()[4][:-1])

                        else:
#                            print("here: ", line.split()[0][1:])
                            tn = int(str(line).split()[0][1:])
                            fp = int(line.split()[1])
                            fn = int(line.split()[2])
                            tp = int(line.split()[-1][:-1])


#                        print(line)
#                        print(tn)

                        precision = round((tp/(tp+fp)), 3)
                        sensitifity = round((tp/(tp+fn)), 3)
                        specificity = round((tn/(tn+fp)), 3)
                        fscore = round((2*((precision * sensitifity)/(precision + sensitifity))), 3)

                    except ValueError as error:
                        print(error, file, '\t', line)
                        print(f"{tn}, {fp}, {fn}, {tp}")
                        print(line.split()[1], line.split()[2], line.split()[3], line.split()[4][:-1])
                        print(type(int(line.split()[1])))

                    except ZeroDivisionError as e:
                        precision, sensitifity, specificity, fscore = "NA", "NA", "NA", "NA"
                        print(e, file)

                    flag = False

                if line.startswith("tn, fp, fn, tp"):
                    flag = True

                if line.startswith("AUC score"):
                    auc = line.split()[-1]

#            print(train_loss)
            wf.write(f"{lr}, {optimizer}, {dropout}, {h1}, {h2}, ,{train_loss}, {valid_loss}, , \
                     {train_acc}, {valid_acc}, {test_acc}, , \
                     {train_f1}, {valid_f1}, , \
                     {precision}, {tp}, {fn}, {sensitifity}, {fp}, {tn}, {specificity}, {fscore}, {epoch}, {auc}\n")
            
            train_loss, valid_loss, train_acc, valid_acc, test_acc, train_f1, valid_f1 = 0, 0, 0, 0, 0, 0, 0 
            precision, tp, fn, sensitifity, fp, tn, specificity, fscore, epoch, auc = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

#    break

'''
