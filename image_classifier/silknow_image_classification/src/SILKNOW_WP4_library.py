# -*- coding: utf-8 -*-
"""
"""
import os
import collections
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
import tensorflow_hub as hub
import xlsxwriter
#import tensorflow_probability as tfp

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score

# also needed (but not in a function) in train_ResNet_152_V2_MTL.py
def master_file_to_collections_list(master_dir, master_file_name):
    r"""Imports the collection names from the master file.
    
    :Arguments\::
        :master_dir (*string*)\::
            This variable is a string and contains the name of the master file.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" have to be in the same folder as the master
            file.
            In the "collection.txt" are relative paths to the images and the
            according class label listed. The paths in a "collection.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The "collection.txt" has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
        :master_file_name (*string*)\::
            The name of the 
    
    :Returns\::
        :collections_list (*list*)\::
            Contains as entries the collection.txt as a string.
            Each array entry corresponds to one collection.
            In the collection.txt are relative paths to the images and the
            according class label listed. The paths in a collection.txt has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg'.
            The collection.txt has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
    """
    # Get image_lists out of the Master.txt
    master_id = open(os.path.abspath(master_dir + '/' + master_file_name), 'r')
    collections_list = []
    for collection in master_id:
        collections_list.append(collection.strip())
    master_id.close()
#    print('Got the following collections:', collections_list, '\n')
    return collections_list


def get_image_paths_from_collection(master_dir, collections_file,
                                    num_input_images):
    r"""Imports the image paths from a collection.
    
    :Arguments\::
        :master_dir (*string*)\::
            This variable is a string and contains the name of the master file.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" have to be in the same folder as the master
            file.
            In the "collection.txt" are relative paths to the images and the
            according class label listed. The paths in a "collection.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The "collection.txt" has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
        :collections_file (*string*)\::
            In the "collection.txt" are relative paths to the images and the
            according class label listed. The paths in a "collection.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The "collection.txt" has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
        :num_input_images (*int*)\::
            The number of input images that shall be considered.
    
    :Returns\::
        :image_path_list (*list*)\::
            It's a list containing the absolute paths of the num_input_images
            images.
    """
    coll_id = open(os.path.join(master_dir, collections_file), 'r')
    image_path_list = []
    first_line_passed = False
    image_counter = 0
    for im_line in coll_id:
        if first_line_passed == False:
            first_line_passed = True
            continue
        if image_counter == num_input_images:
            break  
        image_path  = im_line.split('\t')[0]
        image_path_list.append(os.path.join(master_dir, image_path))
        image_counter = image_counter + 1
    
    coll_id.close()
    
    if image_counter < num_input_images:
        print('Desired number of images for that similar images',
              'shall be searched:', num_input_images)
        print('Number of provided images:', image_counter)
        print('Continuing with the given', image_counter, 'images')
    return image_path_list


# from: train_ResNet_152_V2_MTL.py
def collections_list_to_image_lists(collections_list, labels_2_learn, master_dir, multiLabelsListOfVariables=None,
                                    bool_unlabeled_dataset=None):
    r"""Creates image_lists.
    
    :Arguments\::
        :collections_list (*list*)\::
            Is an array. Contains as entries the collection.txt as a string.
            Each array entry corresponds to one collection.
            In the collection.txt are relative paths to the images and the
            according class label listed. The paths in a collection.txt has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg'.
            The collection.txt has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
        :labels_2_learn (*list*)\::
            The name of the '#label' that shall be learnt. It is of special
            interest if more than one label is given per image in the
            collections.txt.
        :master_dir (*string*)\::
            This variable is a string and contains the name of the master file.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" have to be in the same folder as the master
            file.
            In the "collection.txt" are relative paths to the images and the
            according class label listed. The paths in a "collection.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The "collection.txt" has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
            
    :Returns\::
        :collections_dict (*dictionary*)\::
            It's a dict, that has the class labels as keys and the according
            paths (relative to the master_file.txt) including the image names
            as values.
        :label_2_image (*dictionary*)\::
            It's a dict, that has the class labels as keys and the according
            image names as values.
            (Not corrected if class contains too less images!!!)
    
    """
    # Teste, dass gleiche labels in allen collections.txt
    collections_dict = {}
    label_2_image    = {}
    for collection in collections_list:
        coll_id = open(os.path.join(master_dir, collection), 'r', encoding='utf8', errors='ignore')
        first_line_passed = False
        for im_line in coll_id:
            if first_line_passed == False:
                ind_label_2_extract = im_line.strip().replace(' ', '')\
                                    .replace('\t', '').split('#')[1:]\
                                    .index(labels_2_learn[0])
#                label = im_line.strip().replace(' ', '')\
#                                    .replace('\t', '')\
#                                    .split('#')[1:][ind_label_2_extract]
#                print('The following label will be extracted for all images:',
#                      label,
#                      '\n')
                first_line_passed = True
                continue
            
            image_path  = im_line.split('\t')[0]
            class_label = im_line.split('\t')[ind_label_2_extract].replace('\n', '')

            if multiLabelsListOfVariables is None:
                if "___" in class_label:
                    class_label = 'nan'
            else:
                if labels_2_learn[0] not in multiLabelsListOfVariables and "___" in class_label:
                    class_label = 'nan'

            image_name  = image_path.split('/')[-1]
            
            if class_label not in collections_dict.keys():
                collections_dict[class_label] = [image_path]
            else:
                collections_dict[class_label].append(image_path)
                
            if class_label not in label_2_image.keys():
                label_2_image[class_label] = [image_name]
            else:
                label_2_image[class_label].append(image_name)
        coll_id.close()

    if 'NaN' in collections_dict.keys() and bool_unlabeled_dataset is None:
        del collections_dict['NaN']
    if 'nan' in collections_dict.keys() and bool_unlabeled_dataset is None:
        del collections_dict['nan']
            
    collections_dict = collections.OrderedDict(sorted(collections_dict.items()))
#    print('The following classes will be learnt for the label',
#          label, ':', collections_dict.keys(), '\n')            
    
    return (collections_dict, label_2_image)


# from: train_ResNet_152_V2_MTL.py
def collections_list_MTL_to_image_lists(collections_list, labels_2_learn, master_dir,
                                        multiLabelsListOfVariables=None,
                                        bool_unlabeled_dataset=None):
    r"""Creates image_lists.
    
    :Arguments\::
        :collections_list (*list*)\::
            Is an array. Contains as entries the collection.txt as a string.
            Each array entry corresponds to one collection.
            In the collection.txt are relative paths to the images and the
            according class label listed. The paths in a collection.txt has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg'.
            The collection.txt has to have a Header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
        :labels_2_learn (*list*)\::
            The name of the '#label' that shall be learnt. It is of special
            interest if more than one label is given per image in the
            collections.txt.
        :master_dir (*string*)\::
            This variable is a string and contains the name of the master file.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" have to be in the same folder as the master
            file.
            In the "collection.txt" are relative paths to the images and the
            according class label listed. The paths in a "collection.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The "collection.txt" has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
            
    
    :Returns\::
        :collections_dict_MTL (*dictionary*)\::
            It's a dict containing the different image labels as keys and the
            according "collections_dict" as value.
            
            collections_dict_MTL[#label][class label][images]
        :image_2_label_dict (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. It's needed for the estimation of the multitask
            loss.
            
            image_2_label_dict[full_image_name][variable_name][class_name]
    """
    collections_dict_MTL = {}
    image_2_label_dict   = {}
    label_2_image_dict   = {}
    for im_label in labels_2_learn:
        temp_label_dict, temp_im2label_dict = collections_list_to_image_lists(collections_list,
                                                                              [im_label],
                                                                              master_dir,
                                                                              multiLabelsListOfVariables,
                                                                              bool_unlabeled_dataset)
        collections_dict_MTL[im_label] = temp_label_dict
        label_2_image_dict = {**label_2_image_dict, **temp_im2label_dict}

    # if a label is missing, image_2_label_dict will contain 'nan'
    for im_label in collections_dict_MTL.keys():
        for class_label in collections_dict_MTL[im_label]:
            for image in collections_dict_MTL[im_label][class_label]:
                if image not in image_2_label_dict.keys():
                    temp_label_dict = {}
                    temp_label_dict[im_label] = [class_label]
                    for im_label_2 in collections_dict_MTL.keys():
                        class_label_known = False
                        for class_label_2 in collections_dict_MTL[im_label_2]:
                            if image in collections_dict_MTL[im_label_2][class_label_2]:
                                temp_label_dict[im_label_2] = [class_label_2]
                                class_label_known = True
                        if not class_label_known:
                            temp_label_dict[im_label_2] = ['nan']
                    full_image_name = os.path.abspath(
                                        os.path.join(master_dir, image))
                    image_2_label_dict[full_image_name] = temp_label_dict
    
    return collections_dict_MTL, image_2_label_dict


# imported from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    r"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
#    plt.rcParams["figure.figsize"] = [8, 8]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def estimate_quality_measures(ground_truth, prediction, list_class_names,
                              prefix_plot, res_folder_name):
    r"""
    """
    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)

    # Confusin Matrix (unnormalized)
    # print("ground_truth:",np.asarray(ground_truth))
    # print("prediction:",np.asarray(prediction))
    conf_mat = confusion_matrix(np.asarray(ground_truth),
                                np.asarray(prediction))
    fig_conf_mat = plt.figure()
    plot_confusion_matrix(conf_mat,
                          classes=list_class_names,
                          title='Confusion matrix')
    #    plt.show()
    fig_conf_mat.savefig(res_folder_name + '/' + prefix_plot +
                         '_Confusion_Matrix.png')

    # Confusin Matrix (normalized)
    conf_mat = confusion_matrix(np.asarray(ground_truth),
                                np.asarray(prediction))
    fig_conf_mat_norm = plt.figure()
    plot_confusion_matrix(conf_mat,
                          classes=list_class_names,
                          normalize=True,
                          title='Normalized confusion matrix')
    #    plt.show()
    fig_conf_mat_norm.savefig(res_folder_name + '/' + prefix_plot +
                              '_Confusion_Matrix_normalized.png')

    # f1-scores
    f1_None = f1_score(np.asarray(ground_truth),
                       np.asarray(prediction),
                       average=None)

    # Precision
    precision_None = precision_score(np.asarray(ground_truth),
                                     np.asarray(prediction),
                                     average=None)

    # Recall
    recall_None = recall_score(np.asarray(ground_truth),
                               np.asarray(prediction),
                               average=None)

    # Write results to file
    data_amount = len(ground_truth)
    overall_accuracy = accuracy_score(np.asarray(ground_truth),
                                      np.asarray(prediction))
    text_file_result = open(res_folder_name + '/' + prefix_plot +
                            "_evaluation_results.txt", "w")
    text_file_result.write("Total amount of data (" + prefix_plot + "): " +
                           str(data_amount) + " images \n \n")
    text_file_result.write("Overall Accuracy: %.2f %% \n \n" %
                           (overall_accuracy * 100))

    maxStringLength = max(list(map(lambda x: len(x), list_class_names+["Average"])))+4
    text_file_result.write("{:{width}}".format("Class",width=maxStringLength))
    text_file_result.write("Precision [%]    Recall [%]    f1-score [%]    Contribution\n")

    for classes_index, class_name in enumerate(list_class_names):
        text_file_result.write("{:{width}}".format(class_name,width=maxStringLength))
        # print(class_name, classes_index)
        # print(precision_None)
        text_file_result.write("{:>{width}.2f}".format(precision_None[classes_index] * 100,width=len("Precision [%]")))
        text_file_result.write("{:>{width}.2f}".format(recall_None[classes_index] * 100,width=len("Recall [%]")+4))
        text_file_result.write("{:>{width}.2f}".format(f1_None[classes_index] * 100,width=len("f1-score [%]")+4))
        text_file_result.write("{:>{width}}\n".format((ground_truth == classes_index).sum(),
                                                      width=len("Contribution")+4))

    text_file_result.write("\n{:{width}}".format("Average",width=maxStringLength))
    text_file_result.write("{:>{width}.2f}".format(np.mean(precision_None) * 100,width=len("Precision [%]")))
    text_file_result.write("{:>{width}.2f}".format(np.mean(recall_None) * 100,width=len("Recall [%]")+4))
    text_file_result.write("{:>{width}.2f}".format(np.mean(f1_None) * 100,width=len("f1-score [%]")+4))

    text_file_result.close()
    plt.close('all')


def estimate_multi_label_quality_measures(gtvar, prvar, list_class_names, result_dir, multiLabelsListOfVariables,
                                          taskname):
    print(taskname, "\n\n")
    # prepare predictions and groundtruth according to multi-class and multi-label classification
    if multiLabelsListOfVariables is None:
        ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in gtvar])
        prediction = np.squeeze([np.where(pr == list_class_names) for pr in prvar])
        if len(np.unique(ground_truth)) < len(list_class_names):
            list_class_names = [name for name in list_class_names if
                                list(list_class_names).index(name) in np.unique(ground_truth)]
        estimate_quality_measures(ground_truth=ground_truth,
                                  prediction=prediction,
                                  list_class_names=list(list_class_names),
                                  prefix_plot=taskname,
                                  res_folder_name=result_dir)
    else:
        if taskname in multiLabelsListOfVariables:
            task_dict = {taskname: list_class_names}
            create_multi_label_rectangle_confusion_matrix(groundtruth=gtvar,
                                                          predictions=prvar,
                                                          taskname=taskname,
                                                          task_dict=task_dict,
                                                          result_dir=result_dir)
            gt_binary_no_nan = []
            pr_binary_no_nan = []
            gt_binary_all = []
            pr_binary_all = []
            for gt, pr in zip(gtvar, prvar):
                if "nan_OR_" in pr:
                    gt_binary = [1 if temp_class in gt.split("___") else 0 for temp_class in list_class_names]
                    pr_binary = [1 if temp_class == pr.replace("nan_OR_", "") else 0 for temp_class in
                                 list_class_names]
                    gt_binary_all.append(gt_binary)
                    pr_binary_all.append(pr_binary)
                else:
                    gt_binary = [1 if temp_class in gt.split("___") else 0 for temp_class in list_class_names]
                    pr_binary = [1 if temp_class in pr.split("___") else 0 for temp_class in list_class_names]

                    gt_binary_no_nan.append(gt_binary)
                    pr_binary_no_nan.append(pr_binary)
                    gt_binary_all.append(gt_binary)
                    pr_binary_all.append(pr_binary)

            pred_no_nan_whole_var = []
            gt_no_nan_whole_var = []
            pred_all_whole_var = []
            gt_all_whole_var = []
            for class_ind, class_name in enumerate(list_class_names):
                if len(gt_binary_no_nan) > 0:
                    ground_truth = np.asarray(gt_binary_no_nan)[:, class_ind]
                    prediction = np.asarray(pr_binary_no_nan)[:, class_ind]
                    if np.sum(ground_truth) + np.sum(prediction) > 1:
                        estimate_quality_measures(ground_truth=ground_truth,
                                                  prediction=prediction,
                                                  list_class_names=list(
                                                      ["no_" + class_name, class_name]),
                                                  prefix_plot=taskname + "_binary_" + class_name,
                                                  res_folder_name=result_dir)
                        pred_no_nan_whole_var.append(prediction)
                        gt_no_nan_whole_var.append(ground_truth)
                    else:
                        print("(binary) ground truth and predictions do not contain the class: ", class_name)
                else:
                    print("no evaluation for the binary classification of", class_name, "for the variable",
                          taskname,
                          "possible, as there are no predictions for no class of that variable; "
                          "all sigmoid activation were smaller than the selected threshold.")

                ground_truth_all = np.asarray(gt_binary_all)[:, class_ind]
                prediction_all = np.asarray(pr_binary_all)[:, class_ind]
                if np.sum(ground_truth_all) + np.sum(prediction_all) > 1:
                    estimate_quality_measures(ground_truth=ground_truth_all,
                                              prediction=prediction_all,
                                              list_class_names=list(["no_" + class_name, class_name]),
                                              prefix_plot=taskname + "_binary_" + class_name + "_all",
                                              res_folder_name=result_dir)
                    pred_all_whole_var.append(prediction_all)
                    gt_all_whole_var.append(ground_truth_all)
                else:
                    print("(binary all) ground truth and predictions do not contain the class: ", class_name)

            estimate_quality_measures(ground_truth=np.hstack(gt_no_nan_whole_var),
                                      prediction=np.hstack(pred_no_nan_whole_var),
                                      list_class_names=list(["not_class", "class"]),
                                      prefix_plot=taskname + "_binary_whole_var",
                                      res_folder_name=result_dir)
            estimate_quality_measures(ground_truth=np.hstack(gt_all_whole_var),
                                      prediction=np.hstack(pred_all_whole_var),
                                      list_class_names=list(["not_class", "class"]),
                                      prefix_plot=taskname + "_binary_whole_var_all",
                                      res_folder_name=result_dir)
        else:
            ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in gtvar])
            prediction = np.squeeze([np.where(pr == list_class_names) for pr in prvar])
            estimate_quality_measures(ground_truth=ground_truth,
                                      prediction=prediction,
                                      list_class_names=list(list_class_names),
                                      prefix_plot=taskname,
                                      res_folder_name=result_dir)


def get_statistic_dict(num_labels, coll_dict, relevant_labels):
    r"""Creates statistic_dict.
    
    :Arguments\::
        :num_labels (*int*)\::
            The number of labels/tasks/variables that shall have a class label
            for each image that shall be considered in statistic_dict.
        :coll_dict (*dictionary*)\::
            It's a dict, that has the class labels as keys and the according
            paths (relative to the master_file.txt) including the image names
            as values.
        :relevant_labels (*list*)\::
            A list of labels/variables/tasks that shall be considered in the
            creation of the statisitc dict.
            
    :Returns\::        
        :statistic_dict (*dictionary*)\::
            A dictionary having the relevant_labels as keys and lists of the
            according class labels of the found images with num_lables labels
            as values.
        :count_images (*int*)\::
            The number of images that were found with num_labels labels.
    
    """
    count_images = 0
    statistic_dict = {}
    for image in coll_dict.keys():
        label_count = 0
        for variable in coll_dict[image].keys():
            if variable in relevant_labels\
            and coll_dict[image][variable] != 'NaN':
                label_count = label_count + 1
        if label_count == num_labels:
            count_images = count_images + 1
            for variable in coll_dict[image].keys():
                if variable not in statistic_dict.keys():
                    statistic_dict[variable] = [coll_dict[image][variable][0]]
                else:
                    statistic_dict[variable].append(coll_dict[image][variable][0])
    
    return statistic_dict, count_images


def make_label_statistics(class_label_list, output_path, output_file,
                          label):
    full_fig_name = os.path.join(output_path, output_file + '.png')
    full_txt_name = os.path.join(output_path, output_file + '.txt')
    classes = list(set(class_label_list))
    counts  = []
    
    txt_id = open(full_txt_name, 'w')
    for temp_class in classes:
        temp_count = class_label_list.count(temp_class)
        counts.append(temp_count)
        txt_id.write(str(temp_count) + \
                     '\t' + temp_class + '\n')
    txt_id.close()
    
    fig_id    = plt.figure()
    x_entries = np.arange(len(classes))
    plt.bar(x_entries, counts)
    plt.xticks(x_entries, classes, rotation=-45)
    plt.ylabel('number of samples')
    plt.title('Distribution of class labels\n(' + label +')')
    plt.show()
    fig_id.savefig(full_fig_name) 
    
# adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_cooccurence_matrix(cm, classes,
                          title,
                          cmap=plt.cm.Blues,
                          result_folder=None,
                          figsize=(15, 15),
                          dpi=80,
                          fontsize=20):
    r"""
    This function plots the cooccurence matrix.
    """
    
#    plt.rcParams["figure.figsize"] = [8, 8]
    fig_id = plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': fontsize})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.margins(5)
    plt.subplots_adjust(bottom=0.15)
    
    if result_folder is not None:
        fig_id.savefig(os.path.join(result_folder, title + 'png'))
    
def get_cooccurence_statistic(statistic_dict, coll_dict, result_folder=None):
    r"""Plots one cooccurence matrix for each possible level of incompleteness.
    
    :Arguments\::
        :statistic_dict (*dictionary*)\::
            A dictionary having the relevant_labels as keys and lists of the
            according class labels of the found images with num_lables labels
            as values. (i.e. the first output from get_statistic_dict)
        :coll_dict (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. (i.e. the second output from collections_list_MTL_to_image_lists)
        :result_folder (*string*)\::
            Relative path to the folder where the plots are saved. 
            The plots will not be saved if this variable is not passed to the function.
            
    :Returns\::     
    """
    
    # Get names of all occuring class names of all variables
    class_list = []
    for i in range(len(statistic_dict.keys())):
        class_list += list(np.unique(np.asarray(list(statistic_dict.values()))[i]))
    
    # create empty cooccurence matrices
    num_labels = len(statistic_dict.keys())
    coocmat = np.zeros((len(class_list),len(class_list), num_labels), dtype=np.int32)
    
    # fill matrices according to level of incompleteness
    for image in coll_dict.keys():
        image_class_list = np.asarray(list(coll_dict[image].values())).flatten()
        indices = [class_list.index(image_class_list[i]) for i in range(len(image_class_list))]
        for ind1 in indices:
            for ind2 in indices:
                coocmat[ind1, ind2, len(image_class_list)-1] += 1
    # print matrices according to level of incompleteness            
    for incomp_level in range(num_labels):
        plot_cooccurence_matrix(coocmat[:,:,incomp_level], class_list, 
                                title='Cooccurence matrix - '+str(int(incomp_level+1))+' variable(s)', 
                                result_folder=result_folder)
        
    # print cooccurences independent of incompleteness
    plot_cooccurence_matrix(np.sum(coocmat, axis=-1), class_list, 
                            title='Cooccurence matrix - ignoring level of incompleteness', 
                            result_folder=result_folder)


def add_data_augmentation(aug_set_dict, input_im_tensor):
    r"""Realizes data augmentation.

    :Arguments\::
        :aug_set_dict (*dictionary*)\::
            The keys of aug_set_dict are the transformations that will be applied
            during data augmentation, the values are the corresponding transformation
            parameters. The options for augmentations are the following:

            :flip_left_right (*bool*)\::
            Whether to randomly mirrow the image horizontally along its
            vertical centre line.
            :flip_up_down (*bool*)\::
                Whether to randomly mirrow the image vertically along its
                horizontal centre line.
            :random_shear (*list*)\::
                The list contains float value ranges from which the parameters for
                shearing horizontally (hor) and vertically (vert) will be drawn:
                    [lower bound hor, upper boudn hor, lower bound vert,
                    upper bound vert]
            :random_brightness (*float*)\::
                Multiplies all image channels independently of all pixels by a
                random value out of the range
                [1 - random_brightness/100; 1 + random_brightness/100].
            :random_crop (*list*)\::
                Range of float fractions for centrally cropping the image. The crop fraction
                is drawn out of the provided range [lower bound, upper bound],
                i.e. the first and second values of random_crop.
            :random_rotation (*float*)\::
                Randomly rotates the image counter clockwise by a random angle in
                the range [-random_rotation; +random_rotation]. The angle has to be
                given in radians.
            :random_contrast (*list*)\::
               Adjusts the contrast of an image or images by a random factor.
               random_contrast has to be an array with 2 values. The first value
               is the lower bound for the random factor, the second value is the
               upper bound for the random factor.
            :random_hue (*float*)\::
               Adjusts the hue of RGB images by a random factor. random_hue has to
               be between 0 and 0.5. The hue will be adjusted by a random factor
               picked from the interval [-random_hue, random_hue]
            :random_saturation (*list*)\::
               Adjusts the saturation of an image or images by a random factor.
               random_saturation has to be an array with 2 values. The first value
               is the lower bound for the random factor, the second value is the
               upper bound for the random factor.
            :random_rotation90 (*bool*)\::
                If True, the image will be rotated counter-clockwise by 90 degrees
                (with a chance of 50%).
            :gaussian_noise (*float*)\::
                Adds gaussian noise to the image. The noise will be samples from a
                gaussian distribution with zero mean and a standard deviation of
                gaussian_noise.

        :input_im_tensor (*tensor*)\::
            The input image that will be transformed. Must be a tensor of rank 3,
            with sizes [image_height, image_width, image_channels]. The sizes are
            already the ones expected by the network.

    :Returns\::
        : tranformed_image (*tensor*)\::
            The output image after applying the transformations. Is the same type
            as input_im_tensor.

    """
    tranformed_image = input_im_tensor

    if "random_rotation" in aug_set_dict.keys():
        random_rotation = aug_set_dict["random_rotation"]
        # added: random_rotation
        rotation_min  = -random_rotation
        rotation_max =  random_rotation
        rotation_value = tf.random.uniform(shape=[],
                                           minval=rotation_min,
                                           maxval=rotation_max)
        tranformed_image  = tf.contrib.image.rotate(tranformed_image,
                                                 rotation_value)

    if "random_rotation90" in aug_set_dict.keys():
        random_rotation90 = aug_set_dict["random_rotation90"]
    #if random_rotation90 is not None:
        # added: randomly rot90
        if random_rotation90:
            rot90_indicator = tf.random.uniform(shape=[], minval=0, maxval=1)
            test_rot90 = tf.constant(0.5, dtype = tf.float32)
            tranformed_image = tf.cond(rot90_indicator >= test_rot90,
                                    lambda: tf.image.rot90(tranformed_image, k = 1),
                                    lambda: tf.image.rot90(tranformed_image, k = 0))
        else:
            tranformed_image = tranformed_image

    if "flip_left_right" in aug_set_dict.keys():
        flip_left_right = aug_set_dict["flip_left_right"]
    #if flip_left_right is not None:
        if flip_left_right:
            tranformed_image = tf.image.random_flip_left_right(tranformed_image)

    if "flip_up_down" in aug_set_dict.keys():
        flip_up_down = aug_set_dict["flip_up_down"]
    #if flip_up_down is not None:
        # added: random_flip_up_down
        if flip_up_down:
            tranformed_image = tf.image.random_flip_up_down(tranformed_image)

    if "random_brightness" in aug_set_dict.keys():
        random_brightness = aug_set_dict["random_brightness"]
    #if random_brightness is not None:
        # brightness
        brightness_min   = 1.0 - (random_brightness / 100.0)
        brightness_max   = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random.uniform(shape=[],
                                       minval=brightness_min,
                                       maxval=brightness_max)
        tranformed_image = tf.multiply(tranformed_image, brightness_value)

    if "random_contrast" in aug_set_dict.keys():
        random_contrast = aug_set_dict["random_contrast"]
    #if random_contrast is not None:
        # added: random_contrast
        # contrast_min >= 0, contrast_min < contrast_max!
        assert random_contrast[0] >= 0, "Minimum Contrast has to be larger than 0!"
        assert random_contrast[1] > random_contrast[0], "Minimum Contrast has to be smaller than Maximum Contrast!"
        contrast_min      = random_contrast[0]
        contrast_max      = random_contrast[1]
        tranformed_image  = tf.image.random_contrast(tranformed_image,
                                               contrast_min,
                                               contrast_max)

    if "random_hue" in aug_set_dict.keys():
        random_hue = aug_set_dict["random_hue"]
    #if random_hue is not None:
        # added: random_hue
        # random_hue in [0, 0.5]!
        hue_value  = random_hue
        tranformed_image = tf.image.random_hue(tranformed_image, hue_value)

    if "random_saturation" in aug_set_dict.keys():
        random_saturation = aug_set_dict["random_saturation"]
    #if random_saturation is not None:
        # added: random_saturation
        # saturation_min >= 0, saturation_min < saturation_max!
        assert random_saturation[0] >= 0, "Minimum Saturation has to be larger than 0!"
        assert random_saturation[1] > random_saturation[0], "Maximum Saturation has to be smaller than Minimum Saturation!"
        saturation_min   = random_saturation[0]
        saturation_max   = random_saturation[1]
        tranformed_image  = tf.image.random_saturation(tranformed_image,
                                                saturation_min,
                                                saturation_max)

    if "random_shear" in aug_set_dict.keys():
        random_shear = aug_set_dict["random_shear"]
    #if random_shear is not None:
        # added: random_shear
        shear_w = tf.random.uniform(shape=[],
                                    minval = random_shear[0],
                                    maxval = random_shear[1])
        shear_h = tf.random.uniform(shape=[],
                                    minval = random_shear[2],
                                    maxval = random_shear[3])
        trafo_matrix = [1, shear_w, 0,
                        shear_h, 1, 0,
                        0, 0]
        tranformed_image = tf.contrib.image.transform(tranformed_image,
                                             trafo_matrix)

    if "random_crop" in aug_set_dict.keys():
        random_crop = aug_set_dict["random_crop"]
    #if random_crop is not None:
        # added: random_crop
        crop_fraction = random.uniform(random_crop[0], random_crop[1])
        tranformed_image = tf.image.central_crop(tranformed_image, crop_fraction)
        if crop_fraction < 1:
            # resizing to the network input dimensions (bilinear interpolation)
            input_height = tf.shape(input_im_tensor)[0]
            input_width = tf.shape(input_im_tensor)[1]
            resize_shape = tf.stack([input_height, input_width])
            resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
            resized_image_tensor = tf.image.resize(tranformed_image,
                                                          resize_shape_as_int)
            tranformed_image = resized_image_tensor

    if "gaussian_noise" in aug_set_dict.keys():
        gaussian_noise = aug_set_dict["gaussian_noise"]
    #if gaussian_noise is not None:
        noise = tf.random.normal(shape=tf.shape(tranformed_image), mean=0.0, stddev=gaussian_noise, dtype=tf.float32)
        tranformed_image = tf.clip_by_value(tranformed_image + noise, 0.0, 1.0)

    return tranformed_image


def add_jpeg_decoding(module_spec, crop_aspect_ratio=1):
    r"""Adds operations that perform JPEG decoding and resizing to the graph.

        :Arguments\::
            :input_height (*int*)\::
                Height of the input image. Can be ignored if a hub module is used.
            :input_width (*int*)\::
                Width of the input image. Can be ignored if a hub module is used.
            :input_depth (*int*)\::
                Channels of the input image. Can be ignored if a hub module is used.
            :module_spec (*module*)\::
                The hub.ModuleSpec for the image module being used.
            :crop_aspect_ratio (*float*)\::
                If greater than 1, images will be cropped to their central square when their aspect ratio is
                greater than crop_aspect_ratio.
                For example, an image with a height of 200 and a width of 400 has an aspect ratio of 2.
                When crop_aspect_ratio is set to 1.5, the exemplary image will be cropped to the region
                hmin=0 hmax=200 wmin=100 wmax=300.

        :Returns\::
            :jpeg_data_tensor (*tensor*)\::
                Tensor to feed JPEG data into it.
            :resized_image_tensor (*tensor*)\::
                Tensor that contains the result of the preprocessing steps.
        """

    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)

    print('Input dimensions of the pre-trained network:\n',
          input_height, input_width, input_depth)

    jpeg_data_tensor = tf.compat.v1.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data_tensor,
                                         channels=input_depth)
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_height = tf.cast(tf.shape(decoded_image_as_float)[0], tf.float32)
    decoded_width = tf.cast(tf.shape(decoded_image_as_float)[1], tf.float32)

    # Gaussian filtering before downsizing, we assume that the target image size is square
    sigma = tf.cast(tf.shape(decoded_image)[0] / input_height, tf.float32)
    kernelsize = tf.cast(tf.cast(sigma * 6, tf.int32) + (1 - (tf.cast(sigma * 6,
                                                                      tf.int32) % 2)), tf.float32)
    distr = tf.distributions.Normal(0.0, sigma)
    vals = distr.prob(tf.range(start=-kernelsize,
                               limit=kernelsize + 1,
                               dtype=tf.float32))
    gauss_kernel_2d = tf.einsum('i,j->ij',
                                vals,
                                vals)
    gauss_kernel_2d = gauss_kernel_2d / tf.reduce_sum(gauss_kernel_2d)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    gauss_kernel = tf.tile(gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis],
                           [1, 1, 3, 1])
    pointwise_filter = tf.eye(3, batch_shape=[1, 1])  # does nothing
    bool_downsize = tf.compat.v1.placeholder(tf.bool)
    bool_downsize = tf.cond(input_height < tf.maximum(decoded_height, decoded_width),
                            lambda: True, lambda: False)
    decoded_image_4d = tf.cond(bool_downsize,
                               lambda: tf.nn.separable_conv2d(decoded_image_4d,
                                                              gauss_kernel,
                                                              pointwise_filter,
                                                              strides=[1, 1, 1, 1],
                                                              padding="SAME"),
                               lambda: decoded_image_4d)

    # Crop images to their central square, if their aspect ratio is too large
    decoded_height = tf.cast(tf.shape(decoded_image_4d)[1], tf.float32)
    decoded_width = tf.cast(tf.shape(decoded_image_4d)[2], tf.float32)
    aspect_ratio = tf.minimum(decoded_height / decoded_width,
                              decoded_width / decoded_height)
    bool_crop = tf.cond(aspect_ratio < crop_aspect_ratio, lambda: True, lambda: False)
    crop_size = tf.cast(tf.minimum(decoded_height, decoded_width), tf.float32)
    crop_offset_height = tf.cast(tf.floor(decoded_height / tf.constant(2.) - crop_size / tf.constant(2.)), tf.int32)
    crop_offset_width = tf.cast(tf.floor(decoded_width / tf.constant(2.) - crop_size / tf.constant(2.)), tf.int32)
    cropped_image_4d = tf.image.crop_to_bounding_box(decoded_image_4d,
                                                     crop_offset_height,
                                                     crop_offset_width,
                                                     tf.cast(crop_size - 1., tf.int32),
                                                     tf.cast(crop_size - 1., tf.int32))
    decoded_image_4d = tf.cond(bool_crop,
                               lambda: cropped_image_4d,
                               lambda: decoded_image_4d)

    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image_tensor = tf.compat.v1.image.resize_bilinear(decoded_image_4d,
                                                    resize_shape_as_int)
    resized_image_tensor = tf.squeeze(resized_image_tensor)

    return jpeg_data_tensor, resized_image_tensor

def create_multi_label_rectangle_confusion_matrix(groundtruth, predictions, taskname, task_dict, result_dir):
    list_of_possible_classes = list(np.unique(groundtruth))
    base_classes = task_dict[taskname]
    for combi_len in range(len(base_classes)):
        class_combis = list(itertools.combinations(base_classes, combi_len+1))
        for combi in class_combis:
            pot_class = "___".join(combi)
            if pot_class not in list_of_possible_classes:
                list_of_possible_classes.append(pot_class)

    conf_mat = np.zeros((len(list_of_possible_classes), len(list_of_possible_classes)))
    conf_mat_all = np.zeros((len(list_of_possible_classes), len(list_of_possible_classes)))

    for gt_class in np.unique(groundtruth):
        row = list_of_possible_classes.index(gt_class)
        cur_pred_for_gt = predictions[groundtruth == gt_class]
        preds, counts = np.unique(cur_pred_for_gt, return_counts=True)
        for pre, cou in zip(preds, counts):
            # print(pre)
            if "nan_OR_" in pre:
                if pre.replace("nan_OR_","") == "nan":
                    continue
                col = list_of_possible_classes.index(pre.replace("nan_OR_",""))
                conf_mat_all[row, col] +=cou
            else:
                col = list_of_possible_classes.index(pre)
                conf_mat[row, col] += cou
                conf_mat_all[row, col] += cou

    write_multi_label_eval_statistics_to_file(conf_mat, list_of_possible_classes, result_dir, taskname, "_var")
    write_multi_label_eval_statistics_to_file(conf_mat_all, list_of_possible_classes, result_dir, taskname, "_var_all")


def write_multi_label_eval_statistics_to_file(conf_mat, list_of_possible_classes, result_dir, taskname, file_substr):
    workbook = xlsxwriter.Workbook(os.path.join(result_dir, taskname + file_substr + '.xlsx'))
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    # write label names to worksheet
    for column, label in enumerate(list_of_possible_classes):
        worksheet.write(0, column + 1, str(label))
        worksheet.write(0, column + 2, "Recall", bold)
    row_excel = 1
    precision = []
    recall = []
    true_positives = []
    for column, label in enumerate(list_of_possible_classes):
        if np.sum(conf_mat[column, :]) != 0:
            true_positives.append(conf_mat[column, column])

            temp_precision = conf_mat[column, column] / sum(conf_mat[:, column])
            precision.append(temp_precision)

            temp_recall = conf_mat[column, column] / sum(conf_mat[column, :])
            recall.append(temp_recall)
            worksheet.write(row_excel, 0, str(label))
            for conf_ind, conf_entry in enumerate(conf_mat[column, :]):
                worksheet.write(row_excel, conf_ind + 1, int(conf_entry))
            worksheet.write(row_excel, conf_ind + 2, str(np.round(temp_recall * 100, 1)) + "%")

            row_excel += 1
    all_f_scores = []
    worksheet.write(row_excel + 1, 0, "F1-score", bold)
    worksheet.write(row_excel, 0, "Precision", bold)
    for prec_ind, prec_entry in enumerate(precision):
        worksheet.write(row_excel, prec_ind + 1, str(np.round(prec_entry * 100, 1)) + "%")
        f_score = 2 * precision[prec_ind] * recall[prec_ind] / (precision[prec_ind] + recall[prec_ind])
        worksheet.write(row_excel + 1, prec_ind + 1, str(np.round(f_score * 100, 1)) + "%")
        if str(f_score)== "nan":
            all_f_scores.append(0.)
        else:
            all_f_scores.append(f_score)

    worksheet.write(row_excel + 3, 0, "Average F1-score", bold)
    worksheet.write(row_excel + 3, 1, str(np.round(np.mean(all_f_scores) * 100, 1)) + "%")
    worksheet.write(row_excel + 4, 0, "Overall accuracy", bold)
    worksheet.write(row_excel + 4, 1, str(np.round(sum(true_positives) / np.sum(conf_mat)*100, 1)) + "%")
    workbook.close()