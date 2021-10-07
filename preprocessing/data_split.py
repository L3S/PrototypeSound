import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def read_meta(meta_csv):
    df = pd.read_csv(meta_csv, sep=',')
    df = pd.DataFrame(df)

    audio_names = []
    set_categories = []
    cycle_labels = []

    for row in df.iterrows():
        audio_name = row[1]['audio_name']
        set_category = row[1]['set_category']
        cycle_label = row[1]['cycle_label']

        audio_names.append(audio_name)
        set_categories.append(set_category)
        cycle_labels.append(cycle_label)

    return audio_names, set_categories, cycle_labels

def datasplit(meta_dir):
    """
    Split data into train and dev (test set is already defined in ICBHI)
    Args:
        meta_dir: meta data folder
    Returns:
    train.csv, dev.csv, test.csv, traindev.csv
    """
    audio_names, set_categories, cycle_labels = read_meta(os.path.join(meta_dir, 'meta.csv'))

    audio_traindev = []
    audio_test = []
    label_traindev = []
    label_test = []
    subjectid_traindev = []
    subjectid_test = []
    for i in range(0, len(audio_names)):
        if set_categories[i] == 'train':
            audio_traindev.append(audio_names[i])
            label_traindev.append(cycle_labels[i])
            subjectid_traindev.append(audio_names[i].split('_')[0])
        elif set_categories[i] == 'test':
            audio_test.append(audio_names[i])
            label_test.append(cycle_labels[i])
            subjectid_test.append(audio_names[i].split('_')[0])
        else:
            print('Wrong set category!')

    subid_traindev_unique, subid_traindev_unique_ind = np.unique(subjectid_traindev, return_index=True)
    subid_traindev_unique_label = np.array(label_traindev)[subid_traindev_unique_ind]
    '''
    labels, counts = np.unique(subid_traindev_unique_label, return_counts=True)
    print(labels)
    print(counts)
    '''
    stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=12)
    for train_idx, test_idx in stratSplit.split(subid_traindev_unique, subid_traindev_unique_label):
        subid_train_unique, subid_train_unique_label = subid_traindev_unique[train_idx], subid_traindev_unique_label[train_idx]
        subid_dev_unique, subid_dev_unique_label = subid_traindev_unique[test_idx], subid_traindev_unique_label[test_idx]
        '''
        labels, counts = np.unique(subid_train_unique_label, return_counts=True)
        print(labels)
        print(counts)
        labels, counts = np.unique(subid_dev_unique_label, return_counts=True)
        print(labels)
        print(counts)
        '''
    audio_train = []
    audio_dev = []
    label_train = []
    label_dev = []
    for i in range(0, len(subid_train_unique)):
        ind = np.argwhere(np.array(subjectid_traindev) == subid_train_unique[i])
        ind = [j[0] for j in ind]
        audio_train.extend(np.array(audio_traindev)[ind])
        label_train.extend(np.array(label_traindev)[ind])

    for i in range(0, len(subid_dev_unique)):
        ind = np.argwhere(np.array(subjectid_traindev) == subid_dev_unique[i])
        ind = [j[0] for j in ind]
        audio_dev.extend(np.array(audio_traindev)[ind])
        label_dev.extend(np.array(label_traindev)[ind])

    df = pd.DataFrame(data={'audio_name': audio_train, 'cycle_label': label_train})
    df.to_csv(os.path.join(meta_dir, 'meta_train.csv'), index=False)

    df = pd.DataFrame(data={'audio_name': audio_dev, 'cycle_label': label_dev})
    df.to_csv(os.path.join(meta_dir, 'meta_dev.csv'), index=False)

    df = pd.DataFrame(data={'audio_name': audio_test, 'cycle_label': label_test})
    df.to_csv(os.path.join(meta_dir, 'meta_test.csv'), index=False)

    df = pd.DataFrame(data={'audio_name': audio_traindev, 'cycle_label': label_traindev})
    df.to_csv(os.path.join(meta_dir, 'meta_traindev.csv'), index=False)

if __name__ == '__main__':
    meta_dir = '../data_experiment/meta_data/'

    datasplit(meta_dir)
