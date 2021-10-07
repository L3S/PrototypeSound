import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       print_confusion_matrix, print_accuracy, print_accuracy_binary,
                       projection, plot_prototypes)
from models_pytorch import move_data_to_gpu, DecisionLevelMaxPooling, ProtoNet
import config

Model = ProtoNet
batch_size = 16
n_proto_per_class = 1

def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]

    loss = F.nll_loss(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy()
    loss = float(loss)
    
    accuracy = calculate_accuracy(targets, predictions, classes_num, average='macro')

    return accuracy, loss

def forward(model, generate_func, cuda):
    """Forward data to a model.
    
    Args:
      model: object
      generate_func: generate function
      cuda: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        # Predict
        model.eval()
        batch_output = model(batch_x)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
        
    return dict

def train(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    validate = args.validate
    iteration_max = args.iteration_max
    proto_form = args.proto_form
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    # Paths
    if validate:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_train.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_dev.csv')
    else:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_traindev.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_test.csv')
        
    models_dir = os.path.join(workspace, 'models', subdir)
    create_folder(models_dir)
    prototype_dir = os.path.join(workspace, 'prototypes', subdir)
    create_folder(prototype_dir)

    # Model
    model = Model(classes_num, n_proto_per_class * classes_num, proto_form)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)
    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(dataset_dir=dataset_dir,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)
    class_weight = generator.calculate_class_weight()
    class_weight = move_data_to_gpu(class_weight, cuda)

    # Optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    train_bgn_time = time.time()

    best_audio_names = [[] for i in range(0, classes_num)]
    best_x_logmel = [[] for i in range(0, classes_num)]
    best_distance = [[] for i in range(0, classes_num)]
    # Train on mini batches
    for (iteration, (batch_x, batch_y, batch_audio_names)) in enumerate(generator.generate_train()):
        # Evaluate
        if iteration % 100 == 0:
            train_fin_time = time.time()

            (tr_acc, tr_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         max_iteration=None,
                                         cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}'.format(tr_acc, tr_loss))

            (va_acc, va_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='evaluate',
                                         max_iteration=None,
                                         cuda=cuda)
                                
            logging.info('va_acc: {:.3f}, va_loss: {:.3f}'.format(va_acc, va_loss))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        #if iteration % 1000 == 0 and iteration > 0:
        if iteration == iteration_max:
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        # Train
        model.train()
        batch_output, batch_x_logmel, batch_distance, batch_similarity, batch_loss_diverse = model(batch_x)
        best_audio_names, best_x_logmel, best_distance = projection(batch_audio_names, batch_y.data.cpu().numpy(),
                                                     batch_x_logmel.data.cpu().numpy(), batch_output.data.cpu().numpy(),
                                                     batch_distance.data.cpu().numpy(),
                                                     best_audio_names, best_x_logmel, best_distance)
        loss = F.nll_loss(batch_output, batch_y, weight=class_weight)
        #loss_d = torch.mean(torch.min(batch_distance, dim=0)[0]) + torch.mean(torch.min(batch_distance, dim=1)[0])
        #mask = torch.ones(batch_similarity.shape, dtype=torch.bool)
        #mask = move_data_to_gpu(mask, cuda)
        #for i in range(0, len(mask)):
        #    mask[i][batch_y[i]] = False
        #loss_contrasive = torch.mean(batch_similarity[:, batch_y])/\
        #                   (torch.mean(batch_similarity[:, batch_y])+torch.mean(torch.masked_select(batch_similarity, mask)))
        #loss_contrasive = (torch.mean(batch_similarity[:, batch_y])+torch.mean(torch.masked_select(batch_similarity, mask))) / torch.mean(batch_similarity[:, batch_y])
        #loss_contrasive = - torch.log(loss_contrasive)

        loss = loss + 0.1 * batch_loss_diverse #+ 0.1 * loss_contrasive
        #print([x.grad for x in optimizer.param_groups[0]['params']])
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Stop learning
        if iteration == iteration_max:
            plot_prototypes(best_audio_names, best_x_logmel, best_distance, prototype_dir)
            break


def inference_validation_data(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    validate = args.validate
    iteration_max = args.iteration_max
    proto_form = args.proto_form
    filename = args.filename
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    # Paths
    if validate:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_train.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_dev.csv')
    else:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_traindev.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_test.csv')

    model_path = os.path.join(workspace, 'models', subdir, 'md_{}_iters.tar'.format(iteration_max))

    # Load model
    model = Model(classes_num, n_proto_per_class * classes_num, proto_form)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param number:')
    print(param_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    #prototype_dir = os.path.join(workspace, 'prototypes_self', subdir)
    #create_folder(prototype_dir)
    #plot_prototypes_self(model, prototype_dir)

    # Predict & evaluate
    # Data generator
    generator = DataGenerator(dataset_dir = dataset_dir,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    generate_func = generator.generate_validate(data_type='evaluate',
                                                shuffle=False)

    # Inference
    dict = forward(model=model,
                   generate_func=generate_func,
                   cuda=cuda)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)
    classes_num = outputs.shape[-1]

    # Evaluate
    confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
            
    class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)
    se, sp, as_score, hs_score = calculate_accuracy(targets, predictions, classes_num, average='binary')

    # Print
    print_accuracy(class_wise_accuracy, labels)
    print_confusion_matrix(confusion_matrix, labels)
    #print('confusion_matrix: \n', confusion_matrix)
    print_accuracy_binary(se, sp, as_score, hs_score, labels)



if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Example of parser. ')
    
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_dir', type=str, default='../../../data_experiment/')
    parser.add_argument('--subdir', type=str, default='models_dev')
    parser.add_argument('--workspace', type=str, default='../../../experiment_workspace/baseline_cnn_protonet/')
    parser.add_argument('--validate', action='store_true', default=True)
    parser.add_argument('--iteration_max', type=int, default=15000)
    parser.add_argument('--cuda', action='store_true', default=False)
    '''
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--iteration_max', type=int, required=True)
    parser_train.add_argument('--proto_form', type=str, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--validate', action='store_true', default=False)
    parser_inference_validation_data.add_argument('--iteration_max', type=int, required=True)
    parser_inference_validation_data.add_argument('--proto_form', type=str, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    else:
        raise Exception('Error argument!')
