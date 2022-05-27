import os
import numpy as np
import pandas as pd
# import sys

try:
    import silk_classification_class as scc
    import DatasetCreation
    import SILKNOW_WP4_library as wp4lib
except:
    from . import silk_classification_class as scc
    from . import DatasetCreation
    from . import SILKNOW_WP4_library as wp4lib

def create_dataset_parameter(csvfile,
                             imgsavepath,
                             master_file_dir,
                             minnumsamples=150,
                             retaincollections=['cer', 'garin', 'imatex', 'joconde', 'mad', 'met',
                                                'mfa', 'mobilier', 'mtmad', 'paris-musees', 'risd',
                                                'smithsonian', 'unipa', 'vam', 'venezia', 'versailles'],
                             num_labeled=1,
                             multi_label_variables=["material"]):
    """Creates a dataset

    :Arguments\::
        :csvfile (*string*)\::
            The name (including the path) of the CSV file containing the data exported from the SILKNOW knowledge graph.
        :imgsavepath (*string*)\::
            The path to the directory that will contain the downloaded images. The original images will be downloaded
            to the folder img_unscaled in that directory and the rescaled images (the smaller side will be 448 pixels)
            will be saved to the folder img. It has to be relative to the current working directory.
        :master_file_dir (*string*)\::
            Directory where the collection files and masterfile will be created. The storage location can now be chosen
            by the user.
        :minnumsamples (*int*)\::
            The minimum number of samples that has to be available for a single class or, in case the parameter
            multi_label_variables is not None, for every class combination for the variables contained in that list.
            The dataset is restricted to class combinations that occur at least minnumsamples times in the dataset
            exported from the knowledge graph. Classes or class combinations with fewer samples will not be considered
            in the generated dataset.
        :retaincollections (*list of strings*)\::
            A list containing the museums/collections in the knowledge graph that shall be considered for the data set
            creation. Data from museums/collections not stated in this list will be omitted. Possible values in the list
            according to EURECOM’s export from the SILKNOW knowledge graph (19.02.2021) are: cer, garin, imatex,
            joconde, mad, met, mfa, mobilier, mtmad, paris-musee, risd, smithsonian, unipa, vam, venezia, versailles.
        :num_labeled (*int*)\::
            A variable that indicates how many labels per sample should be available so that a sample is a valid sample
            and thus, part of the created dataset. The maximum value is 5, as five semantic variables are considered in
            the current implementation of this function. Choosing this maximum number means that only complete samples
            will form the dataset, while choosing a value of 0 means that records without annotations will also be
            considered. The value of num_labeled must not be smaller than 0.
        :multi_label_variables (*list of strings*)\::
            A list of keywords indicating those among the five semantic variables in the input CSV file (see csvfile)
            that may have multiple class labels per variable to be predicted. A complete list would be ["material",
            "place", "timespan", "technique", "depiction"]. If the value is None, all variables will have mutually
            exclusive labels, i.e. the generated dataset will not contain any samples with a class combination as a
            label.

    :Returns\::
        No returns. This function produces all files needed for running the subsequent software.
    """
    DatasetCreation.createDataset(rawCSVFile = csvfile,
                                  imageSaveDirectory=imgsavepath,
                                  masterfileDirectory=master_file_dir,
                                  minNumSamplesPerClass=minnumsamples,
                                  retainCollections=retaincollections,
                                  minNumLabelsPerSample=num_labeled,
                                  flagDownloadImages=True,
                                  flagRescaleImages=True,
                                  fabricListFile=None,
                                  multiLabelsListOfVariables=multi_label_variables)

def train_model_parameter(masterfile_name,
                          masterfile_dir,
                          log_dir,
                          num_finetune_layers=5,
                          relevant_variables=["material", "timespan", "technique", "depiction", "place"],
                          batchsize=300,
                          add_fc = [1024,128],
                          how_many_training_steps=500,
                          how_often_validation=10,
                          validation_percentage=25,
                          learning_rate=1e-3,
                          random_crop=[1., 1.],
                          random_rotation90=False,
                          gaussian_noise=0.0,
                          flip_left_right=False,
                          flip_up_down=False,
                          weight_decay=1e-3,
                          nameOfLossFunction="focal",
                          multi_label_variables=None):
    """Trains a new classifier.

        :Arguments\::
            :masterfile_name (*string*)\::
                Name of the master file that lists the collection files with the available samples that will be used
                for training the CNN. This file has to exist in directory master_dir.
            :masterfile_dir (*string*)\::
                Path to the directory containing the master file.
            :log_dir (*string*)\::
                Path to the directory to which the trained model and the log files will be saved.
            :num_finetune_layers (int)\::
                Number of residual blocks (each containing 3 convo- lutional layers) of ResNet 152 that shall be
                retrained.
            :relevant_variables (list)\::
                A list containing the names of the variables to be learned. These names have to be those (or a subset
                of those) listed in the header sections of the collection files collection_n.txt.
            :batchsize (int)\::
                Number of samples that are used during each training iteration.
            :how_many_training_steps (int)\::
                Number of training iterations.
            :how_often_validation (int)\::
                Number of training iterations between two computations of the validation loss.
            :validation_percentage (int)\::
                Percentage of training samples that are used for validation. The value has to be in the range [0, 100).
            :learning_rate (float)\::
                Learning rate for the training procedure.
            :random_crop (*list*)\::
                Range of float fractions for centrally cropping the image. The crop fraction
                is drawn out of the provided range [lower bound, upper bound],
                i.e. the first and second values of random_crop. If [0.8, 0.9] is given,
                a crop fraction of e.g. 0.85 is drawn meaning that the crop for an image with
                the dimensions 200 x 400 pixels consists of the 170 x 340 central pixels.
            :random_rotation90 (*bool*)\::
                Data augmentation: should rotations by 90° be used (True) or not (False)?
            :gaussian_noise (*float*)\::
                Data augmentation: Standard deviation of the Gaussian noise
            :flip_left_right (*bool*)\::
                Data augmentation: should horizontal flips be used (True) or not (False)?
            :flip_up_down (*bool*)\::
                Data augmentation: should vertical flips be used (True) or not (False)?.
            :weight_decay (float)\::
                Weight of the regularization term in the loss function.
            :nameOfLossFunction (bool)\::
                Indicates the loss function that shall be used:
                    - If "sce": Softmax cross entropy loss for multi-task learning with incomplete samples.
                    (Note: both single-task learning and the complete samples case are special cases of "sce")
                    - If "focal": Focal softmax cross entropy loss for multi-task learning with incomplete samples.
                    (Note: both single-task learning and the complete samples case are special cases of "focal")
                    - If "mixed_sce": Softmax cross entropy loss (for variables listed in relevant_variables, but
                    not in multi_label_variables) combined with Sigmoid cross entropy loss
                    (for variables listed both in relevant_variables and multi_label_variables) for
                    multi-task learning with incomplete samples.
                    (Note: both single-task learning and the complete samples case are special cases of "mixed_sce")
            :multi_label_variables (*list of strings*)\::
                A list of those among the variables to be predicted (cf. relevant_variables) that may have multiple
                class labels per variable to be used in subsequent functions. A complete list would be ["material",
                "place", "timespan", "technique", "depiction"].
            :num_nodes_joint_fc (int)\::
                Number of nodes in each joint fully connected layer.
            :num_finetune_layers (int)\::
                Number of joint fully connected layers.

        :Returns\::
            No returns. This function produces all files needed for running the software.
        """

    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.masterfile_name = masterfile_name
    sc.masterfile_dir = masterfile_dir
    sc.log_dir = log_dir

    sc.add_fc = add_fc
    sc.num_finetune_layers = num_finetune_layers

    sc.relevant_variables = relevant_variables
    sc.batchsize = batchsize
    sc.how_many_training_steps = how_many_training_steps
    sc.how_often_validation = how_often_validation
    sc.validation_percentage = validation_percentage
    sc.learning_rate = learning_rate
    sc.weight_decay = weight_decay
    sc.num_task_stop_gradient = -1
    sc.dropout_rate = 0.1
    sc.nameOfLossFunction = nameOfLossFunction
    sc.lossParameters = {}

    sc.aug_set_dict['random_crop'] = random_crop
    sc.aug_set_dict['random_rotation90'] = random_rotation90
    sc.aug_set_dict['gaussian_noise'] = gaussian_noise
    sc.aug_set_dict['flip_left_right'] = flip_left_right
    sc.aug_set_dict['flip_up_down'] = flip_up_down

    sc.image_based_samples = True
    sc.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sc.train_model()


def classify_images_parameter(masterfile_name,
                              masterfile_dir,
                              model_dir,
                              result_dir,
                              multi_label_variables=None,
                              sigmoid_activation_thresh=0.5):
    """Classifies images.

    :Arguments\::
        :masterfile_name (*string*)\::
            Name of the master file that lists the collection files with the available samples that will be classified
            by the trained CNN in model_dir. This file has to exist in directory master_dir.
        :masterfile_dir (*string*)\::
            Path to the directory containing the master file master_file_name.
        :model_dir (*string*)\::
            Path to the directory with the trained model to be used for the classification. This directory is
            equivalent to log_dir in the function crossvalidation_parameter.
        :result_dir (*string*)\::
            Path to the directory to which the classification results will be saved. This directory is equivalent to
            log_dir in the function crossvalidation_parameter.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the semantic variables that have multiple class labels per variable to be used.
            A complete list would be ["material", "place", "timespan", "technique", "depiction"]. The performed
            classification of variables listed in this parameter is a multi-label classification where one binary
            classification per class is performed. All classes with a sigmoid activation larger than
            sigmoid_activation_thresh are part of the prediction.
            Note that this parameter setting has to be the same as the setting of multi_label_variables in the function
            train_model_parmeter at training time of the CNN that is loaded via model_dir!
        :sigmoid_activation_thresh (*float*)\::
            This variable is a float threshold defining the minimum value of the sigmoid activation in case of a
            multi-label classification that a class needs to have to be predicted. It is 0.5 per default in case that
            the user does not change the value. This parameter is only used, if multi_label_variables is different from
            None.

    :Returns\::
        No returns. This function produces all files needed for running the subsequent software.
    """
    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.masterfile_name = masterfile_name
    sc.masterfile_dir = masterfile_dir
    sc.model_dir = model_dir
    sc.result_dir = result_dir
    sc.bool_unlabeled_dataset = True

    sc.image_based_samples = True
    sc.multiLabelsListOfVariables = multi_label_variables
    sc.sigmoid_activation_thresh=sigmoid_activation_thresh

    # call main function
    sc.classify_images()


def evaluate_model_parameter(pred_gt_dir,
                             result_dir,
                             multi_label_variables=None):
    """Evaluates a model

    :Arguments\::
        :pred_gt_dir (*string*)\::
            Path to the directory where the classification results to be evaluated are saved. This directory is
            equivalent to log_dir in the function crossvalidation_parameter.
        :result_dir (*string*)\::
            Path to the directory to which the evaluation results will be saved. This directory is equivalent to
            log_dir in the function crossvalidation_parameter.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the semantic variables that have multiple class labels per variable to be used.
            A complete list would be ["material", "place", "timespan", "technique", "depiction"]. This list has to be
            identical to the one used in the function classify_model_parmeter at the time of the classification that
            produced the predictions in pred_gt_dir.

    :Returns\::
        No returns.
    """
    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.pred_gt_dir = pred_gt_dir
    sc.result_dir = result_dir
    sc.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sc.evaluate_model()


def postprocess_predictions_for_txt(input_img_centered_pred,
                                   output_dir,
                                   masterfile_dir,
                                   masterfile_name,
                                   model_dir,
                                   path_img_data):
    task_dict = np.load(model_dir + r"/task_dict.npz", allow_pickle=True)["arr_0"].item()
    task_list = task_dict.keys()

    coll_list = wp4lib.master_file_to_collections_list(masterfile_dir, masterfile_name)
    coll_dict, data_dict = wp4lib.collections_list_MTL_to_image_lists(
        collections_list=coll_list,
        labels_2_learn=task_list,
        master_dir=masterfile_dir,
        multiLabelsListOfVariables=[],
        bool_unlabeled_dataset=True)

    for file in os.listdir(input_img_centered_pred):
        if "sys_integration" in file and "post" not in file:
            pass
        else:
            continue

        # imgae-centered prediction file to object-centered prediction file
        pred_df = pd.read_csv(os.path.join(input_img_centered_pred, file))
        final_df = pd.DataFrame(columns=pred_df.columns)
        pred_obj = list(pred_df.obj_uri)
        unique_obj, counts = np.unique(pred_obj, return_counts=True)

        for obj, count in zip(unique_obj, counts):
            if count == 1:
                final_df = final_df.append(pred_df[pred_df.obj_uri == obj], ignore_index=True)
            else:
                temp = pred_df[pred_df.obj_uri == obj]
                max_score = temp[temp[" class_score"] == max(temp[" class_score"])]
                if len(max_score) > 1: max_score = max_score.iloc[0]
                final_df = final_df.append(max_score, ignore_index=True)

        final_df.to_csv(os.path.join(input_img_centered_pred, file.replace(".csv", "_post.csv")), index=False)

        # object centered evaluation
        cur_variable = file.split("_")[-1].split(".")[0]
        prefix_plot = cur_variable + "_obj_cent"
        cur_integration = pd.read_csv(os.path.join(input_img_centered_pred, file))
        all_pred = [pred.strip() for pred in list(cur_integration[' predicted_class'])]
        all_ids = [museum.strip() + "__" + uri.split("/")[-1] + "__" + name.strip() for museum, uri, name in
                   zip(list(cur_integration[' museum']),
                       list(cur_integration['obj_uri']),
                       list(cur_integration[' image_name']))]
        all_gt = [data_dict[os.path.join(os.path.abspath(path_img_data),
                                         cur_id.strip())][cur_variable][0] for cur_id in all_ids]
        nan_mask = np.squeeze(np.asarray(all_gt) != 'nan')
        ground_truth = np.squeeze(np.asarray(all_gt)[nan_mask])
        prediction = np.squeeze(np.asarray(all_pred)[nan_mask])

        ground = np.squeeze([np.where(gt == np.asarray(task_dict[cur_variable])) for gt in ground_truth])
        pred = np.squeeze([np.where(pr == np.asarray(task_dict[cur_variable])) for pr in prediction])

        wp4lib.estimate_quality_measures(ground_truth=ground,
                                         prediction=pred,
                                         list_class_names=task_dict[cur_variable],
                                         prefix_plot=prefix_plot,
                                         res_folder_name=output_dir)