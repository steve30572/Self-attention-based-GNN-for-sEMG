import argparse

import numpy as np
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode
import model.ARMBAND_bias2 as ARMBANDGNN
import model.ConvLSTM as CLSTM
torch.manual_seed(0)
np.random.seed(0)
import warnings
warnings.filterwarnings(action='ignore')

total_loss = []
total_acc = []
valid_loss = []
valid_acc = []


def add_args(parser):
    parser.add_argument('--epoch', type=int, default=150, metavar='N',
                        help='number of training')

    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='insert batch size for training(default 128)')

    parser.add_argument('--precision', type=float, default=1e-6, metavar='N',
                        help='reducing learning rate when a metric has stopped improving(default = 0.0000001')

    parser.add_argument('--channel',default='[24, 16, 8, 4]',metavar='N', help=' 3 channel')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='probability of elements to be zero')
    parser.add_argument('--type', type=int, default=2, metavar='N',
                        help='0: GNN, 1: concat version, 2: GAT version')
    parser.add_argument('--indi', type=int, default=1, metavar='N',
                        help='0: total, 1: indi')
    parser.add_argument('--data', default='example_data.npy', metavar='N',
                        help='name of dataset, nina5_data_xshit.npy, new_data_36.npy, evaluation_example.npy')
    parser.add_argument('--label', default='example_label.npy', metavar='N',
                        help='name of label nina5_label.npy, new_label_36.npy, evaluation_labels.npy')
    parser.add_argument('--cand_num', type=int, default=1, metavar='N',
                        help='number of candidates for each dataset, 10, 36, 17')
    parser.add_argument('--load_data', default='./utils/saved_model/4th_sep4.pt', metavar='N',
                        help='saved model name(no duplicate)')
    parser.add_argument('--num_label', type=int, default= 18, metavar='N',
                        help = 'numbe of label')

    args = parser.parse_args()

    return args


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)

    new_labels, new_examples = [], []
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])

    return new_examples, new_labels


def calculate_fitness(args, examples_training, labels_training, examples_test_0, labels_test_0, examples_test_1,
                      labels_test_1):
    accuracy_test0, accuracy_test1 = [], []
    X_fine_tune_train, Y_fine_tune_train = [], []
    X_test_0, X_test_1, Y_test_0, Y_test_1 = [], [], [], []

    for dataset_index in range(0, 17):
        for label_index in range(len(labels_training)):
            if label_index == dataset_index:
                for example_index in range(len(examples_training[label_index])):
                    if (example_index < 28):
                        X_fine_tune_train.extend(examples_training[label_index][example_index])
                        Y_fine_tune_train.extend(labels_training[label_index][example_index])
        print("{}-th data set open~~~".format(dataset_index))
        for label_index in range(len(labels_test_0)):
            if label_index == dataset_index:
                for example_index in range(len(examples_test_0[label_index])):
                    X_test_0.extend(examples_test_0[label_index][example_index])
                    Y_test_0.extend(labels_test_0[label_index][example_index])

        for label_index in range(len(labels_test_1)):
            if label_index == dataset_index:
                for example_index in range(len(examples_test_1[label_index])):
                    X_test_1.extend(examples_test_1[label_index][example_index])
                    Y_test_1.extend(labels_test_1[label_index][example_index])

    X_fine_tunning, Y_fine_tunning = scramble(X_test_1, Y_test_1)


    valid_examples = X_fine_tunning[0:int(len(X_fine_tunning) * 0.1)]
    labels_valid = Y_fine_tunning[0:int(len(Y_fine_tunning) * 0.1)]
    X_fine_tune = X_fine_tunning[int(len(X_fine_tunning) * 0.1):]
    Y_fine_tune = Y_fine_tunning[int(len(Y_fine_tunning) * 0.1):]
    print("total data size :", len(X_fine_tune_train), np.shape(np.array(X_fine_tune_train)))
    X_fine_tune = torch.from_numpy(np.array(X_fine_tune, dtype=np.float32))

    print("train data :", np.shape(np.array(X_fine_tune)))
    Y_fine_tune = torch.from_numpy(np.array(Y_fine_tune, dtype=np.float32))
    valid_examples = torch.from_numpy(np.array(valid_examples, dtype=np.float32))

    print("valid data :", np.shape(np.array(valid_examples)))
    labels_valid = torch.from_numpy(np.array(labels_valid, dtype=np.float32))
    # dimension setting
    X_test_0 = torch.from_numpy(np.array(X_fine_tune_train, dtype=np.float32))
    Y_test_0 = torch.from_numpy(np.array(Y_fine_tune_train, dtype=np.float32))

    X_test_1 = torch.from_numpy(np.array(X_test_1, dtype=np.float32))


    Y_test_1 = torch.from_numpy(np.array(Y_test_1, dtype=np.float32))
    print(X_test_0.shape, X_test_1.shape, X_fine_tune.shape)
    print(torch.equal(X_test_0, X_test_1), torch.equal(X_fine_tune, X_test_1), torch.equal(X_test_0, X_fine_tune))

    # dataset
    train = TensorDataset(X_fine_tune, Y_fine_tune)
    valid = TensorDataset(valid_examples, labels_valid)
    test_0 = TensorDataset(X_test_0, Y_test_0)
    test_1 = TensorDataset(X_test_1, Y_test_1)

    # data loading
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=False)
    test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=1, shuffle=False)
    test_1_loader = torch.utils.data.DataLoader(test_1, batch_size=1, shuffle=False)

    # model create
    stgcn = ARMBANDGNN.ARMBANDGNN([12, 16, 8, 4], 2, 52)

    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(stgcn.parameters(), lr=0.0035) #lr=args.lr)  # lr=0.0404709 lr=args.lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                     verbose=True, eps=args.precision) #학습이 개선되지 않을때 자동으로 학습률을 조절합니다.
    #training
    model, num_epoch = train_model(stgcn, criterion, optimizer, scheduler,\
                                   {"train": train_loader, "val": valid_loader}, args.epoch, args.precision)
    model.eval()

    # test : set_0
    total = 0
    correct_prediction_test_0 = 0
    for k, data_test_0 in enumerate(test_0_loader):
        inputs_test_0, ground_truth_test_0 = data_test_0
        inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0), Variable(ground_truth_test_0)
        concat_input = inputs_test_0
        for i in range(20):
            concat_input = torch.cat([concat_input, inputs_test_0])

        outputs_test_0 = model(concat_input)
        _, predicted = torch.max(outputs_test_0.data, 1)
        correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
                                      ground_truth_test_0.data.cpu().numpy()).sum()
        total += ground_truth_test_0.size(0)
    accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))
    print("ACCURACY TESƒT_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))

    # test : set_1
    total = 0
    correct_prediction_test_1 = 0
    for k, data_test_1 in enumerate(test_1_loader):
        inputs_test_1, ground_truth_test_1 = data_test_1
        inputs_test_1, ground_truth_test_1 = Variable(inputs_test_1), Variable(ground_truth_test_1)
        concat_input = inputs_test_1
        for i in range(20):
            concat_input = torch.cat([concat_input, inputs_test_1])
        outputs_test_1 = model(concat_input)
        _, predicted = torch.max(outputs_test_1.data, 1)
        correct_prediction_test_1 += (mode(predicted.cpu().numpy())[0][0] ==
                                      ground_truth_test_1.data.cpu().numpy()).sum()
        total += ground_truth_test_1.size(0)
    accuracy_test1.append(100 * float(correct_prediction_test_1) / float(total))
    print("ACCURACY TEST_1 FINAL : %.3f %%" % (100 * float(correct_prediction_test_1) / float(total)))

    #result
    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_test0).mean())
    print("AVERAGE ACCURACY TEST 1 %.3f" % np.array(accuracy_test1).mean())
    return accuracy_test0, accuracy_test1, num_epoch


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, precision):
    since = time.time()
    best_loss = float('inf')
    patience = 60
    patience_increase = 20
    hundred = False

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs#.cuda()
                labels = labels#.cuda()
                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                    outputs = model(inputs) # forward
                    # print(outputs[0][-1].shape)
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:# phase == 'val'
                    model.eval()
                    accumulated_predicted = Variable(torch.zeros(len(inputs), 52))
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(1):#range(20)
                        outputs = model(inputs)
                        labels = labels.long()
                        loss = criterion(outputs, labels)
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            if phase == 'val':
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            else:
                total_loss.append(epoch_loss)
                total_acc.append(epoch_acc)
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val': #'val':  #TODO changed to train
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), args.load_data)
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    print("stopped because of 100%")
                    hundred = True

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience :
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model_weights = torch.load(args.load_data)
    model.load_state_dict(model_weights)
    model.eval()
    return model, num_epochs

def calculate_fitness2_indi(args, examples_training, labels_training):
    accuracy_test0, accuracy_test1 = [], []
    for dataset_index in range(0, args.cand_num):
        X_fine_tune_train, Y_fine_tune_train = [], []
        for label_index in range(len(labels_training)):
            if label_index == dataset_index:
                for example_index in range(len(examples_training[label_index])):
                    # if (example_index < 208):
                        X_fine_tune_train.extend(examples_training[label_index][example_index])
                        Y_fine_tune_train.extend(labels_training[label_index][example_index])
        print("{}-th data set open~~~".format(dataset_index))


        X_fine_tunning, Y_fine_tunning = scramble(X_fine_tune_train, Y_fine_tune_train)


        valid_examples = X_fine_tunning[0:int(len(X_fine_tunning) * 0.1)]
        labels_valid = Y_fine_tunning[0:int(len(Y_fine_tunning) * 0.1)]
        X_fine_tune = X_fine_tunning[int(len(X_fine_tunning) * 0.2):]
        Y_fine_tune = Y_fine_tunning[int(len(Y_fine_tunning) * 0.2):]
        X_test_0 = X_fine_tunning[int(len(X_fine_tunning)*0.1):int(len(X_fine_tunning)*0.2)]
        Y_test_0 = Y_fine_tunning[int(len(X_fine_tunning)*0.1):int(len(X_fine_tunning)*0.2)]

        print("total data size :", len(X_fine_tune_train), np.shape(np.array(X_fine_tune_train)))

        X_fine_tune = torch.from_numpy(np.array(X_fine_tune, dtype=np.float32))
        #X_fine_tune = torch.transpose(X_fine_tune, 1, 2)

        print("train data :", np.shape(np.array(X_fine_tune)))
        Y_fine_tune = torch.from_numpy(np.array(Y_fine_tune, dtype=np.float32))
        valid_examples = torch.from_numpy(np.array(valid_examples, dtype=np.float32))
        #valid_examples = torch.transpose(valid_examples, 1, 2)

        print("valid data :", np.shape(np.array(valid_examples)))
        labels_valid = torch.from_numpy(np.array(labels_valid, dtype=np.float32))
        # dimension setting
        X_test_0 = torch.from_numpy(np.array(X_test_0, dtype=np.float32))
        #X_test_0 = torch.transpose(X_test_0, 1, 2)
        Y_test_0 = torch.from_numpy(np.array(Y_test_0, dtype=np.float32))


        # dataset
        # X_fine_tune = torch.unsqueeze(X_fine_tune, dim=2)
        # valid_examples = torch.unsqueeze(valid_examples, dim=2)
        # X_test_0 = torch.unsqueeze(X_test_0, dim=2)

        train = TensorDataset(X_fine_tune, Y_fine_tune)
        valid = TensorDataset(valid_examples, labels_valid)
        test_0 = TensorDataset(X_test_0, Y_test_0)
        print(torch.unique(Y_fine_tune))
        print(torch.unique(labels_valid))
        print(torch.unique(Y_test_0))
        # data loading
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
        test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=1, shuffle=False)


        stgcn = ARMBANDGNN.ARMBANDGNN(eval(args.channel), args.type, args.num_label)
        print(X_fine_tune.shape)

        # stgcn = CLSTM.ConvLSTM(1, 16, (3, 3), 1)#.cuda()
        precision = 1e-8
        criterion = nn.NLLLoss(size_average=False)
        optimizer = optim.Adam(stgcn.parameters(), lr=args.lr) #lr=args.lr)  # lr=0.0404709 lr=args.lr
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
        #                                                  verbose=True, eps=args.precision)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.00,  verbose=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.4, patience=5,
                                                         verbose=True, eps=precision)
        #training
        model, num_epoch = train_model(stgcn, criterion, optimizer, scheduler,\
                                       {"train": train_loader, "val": valid_loader}, args.epoch, args.precision)
        model.eval()
        all_dict = dict()
        correct_dict = dict()
        for i in range(args.num_label):
            all_dict[i] = 0
            correct_dict[i] = 0

        acc_perlabel = []
        # test : set_0
        total = 0
        correct_prediction_test_0 = 0
        time_list = []
        for k, data_test_0 in enumerate(test_0_loader):
            start_time = time.time()
            inputs_test_0, ground_truth_test_0 = data_test_0
            inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0), Variable(ground_truth_test_0)
            concat_input = inputs_test_0
            for i in range(20): #input data 옆으로 20개 복사 concat
                concat_input = torch.cat([concat_input, inputs_test_0])

            outputs_test_0 = model(concat_input)
            _, predicted = torch.max(outputs_test_0.data, 1)
            all_dict[int(ground_truth_test_0)] += 1
            if mode(predicted.cpu().numpy())[0][0] == ground_truth_test_0.data.cpu().numpy():
                correct_dict[int(ground_truth_test_0)] += 1
            correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
                                          ground_truth_test_0.data.cpu().numpy()).sum()
            total += ground_truth_test_0.size(0)
            end = time.time()
            time_list.append(end-start_time)
        accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))
        print("ACCURACY TESƒT_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))


        # test : set_1
        for i in range(args.num_label):
            try:
                acc_perlabel.append(correct_dict[i]/all_dict[i])
                print("accuracy in %d : %f"%(i, correct_dict[i]/all_dict[i]))
            except:
                continue

        #result
        print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_test0).mean())
        #print("average time: ", np.mean(time_list), time_list)
    return accuracy_test0,  num_epoch

if __name__ == "__main__":
    # loading...
    # examples_training = np.load("../formatted_datasets/evaluation_example.npy", encoding="bytes", allow_pickle=True)
    # labels_training = np.load("../formatted_datasets/evaluation_labels.npy", encoding="bytes", allow_pickle=True)
    # examples_validation0 = np.load("../formatted_datasets/test0_evaluation_example.npy", encoding="bytes",
    #                                allow_pickle=True)
    # labels_validation0 = np.load("../formatted_datasets/test0_evaluation_labels.npy", encoding="bytes",
    #                              allow_pickle=True)
    # examples_validation1 = np.load("../formatted_datasets/test1_evaluation_example.npy", encoding="bytes",
    #                                allow_pickle=True)
    # labels_validation1 = np.load("../formatted_datasets/test1_evaluation_labels.npy", encoding="bytes",
    #                              allow_pickle=True)



    # examples_training = np.load("./EMG_data_for_gestures-master/new_data_1.npy", encoding="bytes", allow_pickle=True)
    # labels_training = np.load("./EMG_data_for_gestures-master/new_label_1.npy", encoding="bytes", allow_pickle=True)
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    examples_training = np.load("./data/" + args.data, encoding="bytes", allow_pickle=True) # data should have shape (# of data, time (24), channel (8), scale (7))
    labels_training = np.load("./data/" + args.label, encoding="bytes", allow_pickle=True)
    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []

    test_0, test_1 = [], []

    for i in range(1):  
        if args.indi == 0:
            accuracy_test_0, num_epochs = calculate_fitness2(args, examples_training, labels_training)
        else:
            accuracy_test_0, num_epochs = calculate_fitness2_indi(args, examples_training, labels_training)
        # accuracy_test_0, accuracy_test_1, num_epochs = calculate_fitness(args, examples_training, labels_training,
        #                                                                  examples_validation0, labels_validation0,
        #                                                                  examples_validation1, labels_validation1)
        print(accuracy_test_0)

        test_0.append(accuracy_test_0)
        # test_1.append(accuracy_test_1)
        print("TEST 0 SO FAR: ", test_0)
        # print("TEST 1 SO FAR: ", test_1)
        print("CURRENT AVERAGE : ", (np.mean(test_0)))

    print("ACCURACY FINAL TEST 0: ", test_0)
    print("ACCURACY FINAL TEST 0: ", np.mean(test_0))
    np.save('training_loss_ours.npy', np.array(total_loss))
    np.save('training_acc_ours.npy', np.array(total_acc))
    np.save('validation_loss_ours.npy', np.array(valid_loss))
    np.save('validation_acc_ours.npy', np.array(valid_acc))

#######
    # parser = argparse.ArgumentParser()
    # args = add_args(parser)
    # examples_training = np.load("./data/CWT_dataset/" + args.data, encoding="bytes", allow_pickle=True)
    # labels_training = np.load("./data/CWT_dataset/" + args.label, encoding="bytes", allow_pickle=True)
    # labels_training -= 8
    # # examples_training = np.expand_dims(examples_training, axis=0)
    # # labels_training = np.expand_dims(labels_training, axis=0)
    # accuracy_one_by_one = []
    # array_training_error = []
    # array_validation_error = []
    #
    # test_0, test_1 = [], []
    # model = ARMBANDGNN_500.ARMBANDGNN(eval(args.channel), args.type, args.num_label)
    # model_weights = torch.load(args.load_data)
    # model.load_state_dict(model_weights)
    # model.eval()
    # correct_prediction_test_0 = 0
    # total = 0
    # for i in range(16):
    #     label_acc = 0
    #     label_total = 0
    #     if i == 6 or i == 10 or i==12:
    #         continue
    #     else:
    #         for j in range(len(labels_training[0,i])):
    #             x = torch.tensor(examples_training[0,i][j], dtype=torch.float32)
    #             x = x.reshape(1, 24, 8, 7)
    #             target = torch.tensor(labels_training[0,i][j], dtype=torch.float32)
    #             y = model(x)
    #             _, predicted = torch.max(y.data, 1)
    #             #ll_dict[int(ground_truth_test_0)] += 1
    #             correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
    #                                           target.data.cpu().numpy()).sum()
    #             total += 1
    #             label_acc += (mode(predicted.cpu().numpy())[0][0] ==
    #                                           target.data.cpu().numpy()).sum()
    #             label_total += 1
    #         print("label %d : %f"%(i, label_acc/label_total))
    # print(correct_prediction_test_0/total)
    # # for i in range(1):  
    # #     if args.indi == 0:
    # #         accuracy_test_0,  num_epochs = calculate_fitness2(args, examples_training, labels_training)
    # #     else:
    # #         accuracy_test_0, num_epochs = calculate_fitness2_indi(args, examples_training, labels_training)
    #
    #
    #
    #
    # #     print(accuracy_test_0)
    # #
    # #     test_0.append(accuracy_test_0)
    # #     #test_1.append(accuracy_test_1)
    # #     print("TEST 0 SO FAR: ", test_0)
    # #     #print("TEST 1 SO FAR: ", test_1)
    # #     print("CURRENT AVERAGE : ", (np.mean(test_0)))
    # #
    # # print("ACCURACY FINAL TEST 0: ", test_0)
    # # print("ACCURACY FINAL TEST 0: ", np.mean(test_0))
    # # #print("ACCURACY FINAL TEST 1: ", test_1)
    # # #print("ACCURACY FINAL TEST 1: ", np.mean(test_1))
    #
    # # with open("Pytorch_results_4_cycles.txt", "a") as myfile:
    # #     myfile.write("stgcn STFT: \n")
    # #     myfile.write("Epochs:")
    # #     myfile.write(str(num_epochs) + '\n')
    # #     myfile.write("Test 0: \n")
    # #     myfile.write(str(np.mean(test_0, axis=0)) + '\n')
    # #     myfile.write(str(np.mean(test_0)) + '\n')
    # #
    # #     myfile.write("Test 1: \n")
    # #     myfile.write(str(np.mean(test_1, axis=0)) + '\n')
    # #     myfile.write(str(np.mean(test_1)) + '\n')
    # #     myfile.write("\n\n\n")

