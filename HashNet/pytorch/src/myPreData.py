# -*- coding: utf-8 -*-

# this file is useful if we have a new dataset and we want to make it work
import numpy as np
import os
import random

random.seed(1)

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def create_file_for_train_by_folderpath(folder_path, file_path, sample_size, select_classes=None,  C=50, full_class_label = True):
    # convert the images orgnized by folders into the txt file that can be used for training
    file_txt =open(file_path, 'w')
    
    if select_classes is None:
        select_classes = np.arange(C)
    
    from publicFunctions import make_one_hot
    subfolder_path_list = os.listdir(folder_path)
    len_subfolder = len(subfolder_path_list)
    replace_i = 0
    for i in range(len_subfolder):
        if i not in select_classes:
            continue
        #if don't want some classes having no sample
        
        
        subfolder_path = folder_path+subfolder_path_list[i]
        image_path_list = os.listdir(subfolder_path)
        len_images = len(image_path_list)
        sample_size_class = sample_size if sample_size<len_images else len_images
        random_train_index = random.sample(range(len_images), sample_size_class)
        if full_class_label:
            label = np.array([i])
            label_one_hot = make_one_hot(label, C).astype(int)
        else:
            label = np.array([replace_i])
            label_one_hot = make_one_hot(label, select_classes.shape[0]).astype(int)
        for j in range(sample_size_class):
            image_index = random_train_index[j]
            image_name = image_path_list[image_index]
            full_path = subfolder_path+'/'+image_name
            
            #print full_path, str(label_one_hot[0]).replace('[','').replace(']','')
            #print (full_path+' '+str(label_one_hot[0]).replace('[','').replace(']','') ).replace('\n','')+'\n'
            file_txt.write( (full_path+' '+str(label_one_hot[0]).replace('[','').replace(']','') ).replace('\n','')+'\n')
        replace_i += 1
    file_txt.close()
            
if __name__ == "__main__":
    data_path = "../data/places365_standard/"
    database_folder_path = data_path+"train/"
    val_folder_path = data_path+"val/"
    
    data_file_path = data_path
    train_file_path = data_path+'train.txt'
    val_file_path = data_path+'val.txt'
    database_file_path = data_path+'database.txt'
    
    doTrain, doTest, doDatabase = False, False, True
    # FXXK the comments! 
    # if want the label's length is C, set it True
    isFullClassLabel = False
    
    folder_path = database_folder_path
    sample_size = 250
    classes_size = 36
    C = 365
    from publicVariables import argmax_ranking_by_class_places365
    select_classes = argmax_ranking_by_class_places365[-classes_size:]
    if doTrain:
        create_file_for_train_by_folderpath(database_folder_path, train_file_path, sample_size, \
                                            select_classes = select_classes, C=C, full_class_label = isFullClassLabel)
    if doTest:
        create_file_for_train_by_folderpath(val_folder_path, val_file_path, sample_size, \
                                    select_classes = select_classes, C=C, full_class_label = isFullClassLabel)
    if doDatabase:
        create_file_for_train_by_folderpath(database_folder_path, database_file_path, 1000, \
                                    select_classes = select_classes, C=C, full_class_label = isFullClassLabel)