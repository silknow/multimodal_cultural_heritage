import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import tensorflow as tf
try:
    import SILKNOW_WP4_library as wp4lib
except:
    from . import SILKNOW_WP4_library as wp4lib

class SampleHandler:
    """Class for handling the usage of samples."""

    def __init__(self, masterfile_dir, masterfile_name, relevant_variables, validation_percentage,
                 image_based_samples=None, sampleType=None, multiLabelsListOfVariables=None,
                 bool_unlabeled_dataset_rules = None, max_num_classes_per_variable_default=None,
                 boolMultiLabelClassification = False):
        """Initializes an object of this class and loads all training and validation images into RAM.

            :Arguments:
              :masterfile_dir (*string*)\::
                  Absolute path to the directory where the masterfile is stored.

              :masterfile_name (*string*)\::
                  Filename of the masterfile.

              :relevant_variables (*list*)\::
                  A list of strings that defines the silk properties that will be considered for classification.

              :image_based_samples (*bool*)\::
                  If true, the samples given in collection files will be interpreted as image-based way, meaning
                  that one sample corresponds to one image.
                  If false, the samples will be interpreted as record-based, meaning that one sample corresponds to
                  one record, which is represented by one or multiple images.

              :validation_percentage (*int*)\::
                  A value between 0 and 100. Defines the percentage of samples that will be used for validation.
                  If this value is 0, no validation will be carried out.

              :sampleType (*string*)\::
                  Indicates the type of samples. There are three options:
                  *image*: Image-based image samples. The same as image_based_samples==True
                  *record*: Record-based image samples. The same as image_based_samples==False
                  *mixed*: Record-based samples with text and image.

              :multiLabelsListOfVariables (*list*)\::
                    List of strings describing fields of variables in the rawCSV that contain multiple labels per
                    variable. All variable names listed in multiLabelsListOfVariables will contain multiple labels per variable in
                    addition to the single variables. Given the labels "label_1", "label_2" and "label_3" for one variable
                    of one image, the resulting collection files will contain a label in the format
                    "label_1___label_2___label_3". Such merged labels will be handeled in subsequent function of the image
                    processing module to perform multi-label classification/semantic similarity.
                """
        assert image_based_samples is not None or sampleType is not None, \
            "Either image_based_samples or sampleType has to be specified!"

        assert sampleType is None or sampleType in ["image", "record", "mixed"], \
            "sampleType has to be one of 'image','record','mixed'!"
        random.seed(42)

        self.masterfile_dir = masterfile_dir
        self.masterfile_name = masterfile_name
        self.validation_percentage = validation_percentage
        self.multiLabelsListOfVariables = multiLabelsListOfVariables
        self.boolMultiLabelClassification = boolMultiLabelClassification

        if sampleType is None:
            self.sampleType = "image" if image_based_samples else "record"
        else:
            self.sampleType = sampleType

        collections_list = wp4lib.master_file_to_collections_list(self.masterfile_dir, self.masterfile_name)
        print('Got the following collections:', collections_list, '\n')

        if self.sampleType == "record" or self.sampleType == "mixed":
            # collectionDataframe is only needed for record-based samples
            self.collectionDataframe = self.getCollectionDataframeFromCollectionsList(collections_list)

        # convert list of collections files into two dictionaries
        # 1) collections_dict_MTL: [variable][class label][images], no explicit 'nan'
        # 2) image_2_label_dict[full_image_name][variable][class_label], with explicit 'nan'
        (collections_dict_MTL,
         image_2_label_dict) = wp4lib.collections_list_MTL_to_image_lists(
            collections_list=collections_list,
            labels_2_learn=relevant_variables,
            master_dir=self.masterfile_dir,
            multiLabelsListOfVariables=self.multiLabelsListOfVariables,
            bool_unlabeled_dataset=bool_unlabeled_dataset_rules)

        self.classCountDict = self.getClassCountDictFromCollectionDict(collections_dict_MTL,
                                                                       bool_unlabeled_dataset_rules)
        self.taskDict = self.getTaskDictFromCollectionDict(collections_dict_MTL)
        self.getDistinctSingleClassesPerTask()

        if not self.multiLabelsListOfVariables is None:

            if bool_unlabeled_dataset_rules is None:
                self.max_num_classes_per_variable = self.find_max_num_classes_per_variable()
            else:
                self.max_num_classes_per_variable = max_num_classes_per_variable_default

        (collections_dict_MTL_train,
         collections_dict_MTL_validation,
         image_2_label_dict_train,
         image_2_label_dict_validation
         ) = self.splitDataIntoTrainingAndValidation(collections_dict_MTL, image_2_label_dict, validation_percentage)

        if bool_unlabeled_dataset_rules is None:
            self.printNumberOfSamplesForEveryClass(collections_dict_MTL_train, collections_dict_MTL_validation)

        self.initializeIndexLists(image_2_label_dict_train, image_2_label_dict_validation)

        self.load_image_data(collections_dict_MTL_train, image_2_label_dict_train,
                             collections_dict_MTL_validation, image_2_label_dict_validation)

    def getDistinctSingleClassesPerTask(self):
        classPerTask = {}
        numClassPerTask = {}
        for task in self.taskDict.keys():
            distinctClasses = []
            for cl in self.taskDict[task]:
                if "___" not in cl:
                    distinctClasses.append(cl)
                else:
                    for singleLabel in cl.split("___"):
                        distinctClasses.append(singleLabel)
            classPerTask[task] = list(np.unique(distinctClasses))
            numClassPerTask[task] = len(np.unique(distinctClasses))
        self.classPerTaskDict = classPerTask
        self.numClassPerTask = numClassPerTask

    def find_max_num_classes_per_variable(self):
        max_num_classes_per_variable = 0
        # print(self.taskDict)
        for task in self.taskDict.keys():
            class_names = []
            for class_label in self.taskDict[task]:
                if "___" in class_label:
                    single_labels = class_label.split("___")
                    for label in single_labels:
                        if label not in class_names:
                            class_names.append(label)
                else:
                    if class_label not in class_names:
                        class_names.append(class_label)
            max_num_classes_per_variable = max(max_num_classes_per_variable, len(class_names))
        return max_num_classes_per_variable

    def initializeIndexLists(self, image_2_label_dict_train, image_2_label_dict_validation):
        # initialize index lists
        self.amountOfTrainingSamples = len(image_2_label_dict_train.keys())
        self.amountOfValidationSamples = len(image_2_label_dict_validation.keys())
        self.indexListTraining = np.arange(0, self.amountOfTrainingSamples)
        self.indexListValidation = np.arange(0, self.amountOfValidationSamples)
        self.nextUnusedTrainingSampleIndex = 0
        self.nextUnusedValidationSampleIndex = 0
        self.epochsCompletedTraining = 0
        self.epochsCompletedValidation = 0
        random.shuffle(self.indexListTraining)
        random.shuffle(self.indexListValidation)

        print("Amount of images for training: %i \n"
              "\nAmount of images for validation: %i \n" % (
                  self.amountOfTrainingSamples, self.amountOfValidationSamples))

    def getClassCountDictFromCollectionDict(self, collections_dict_MTL, bool_unlabeled_dataset_rules):
        # also, check if there is enough data provided
        classCountDict = {}
        for im_label in collections_dict_MTL.keys():
            vardict = {}
            for clsLabel in collections_dict_MTL[im_label].keys():
                temp_class_count = len(collections_dict_MTL[im_label][clsLabel])
                vardict[clsLabel] = temp_class_count
            classCountDict[im_label] = vardict
            if bool_unlabeled_dataset_rules is None:
                assert not len(classCountDict[im_label].keys()) == 0 , \
                    'No valid collections of images found at '
                assert not len(classCountDict[im_label].keys()) == 1, \
                    'Only one class was provided via ' + self.masterfile_name + ' - multiple classes are needed for classification.'
        return classCountDict

    def getCollectionDataframeFromCollectionsList(self, collections_list):
        collectionDataframe = pd.DataFrame()
        for cFile in collections_list:
            df = pd.read_csv(os.path.join(self.masterfile_dir, cFile), delimiter="\t")
            collectionDataframe = collectionDataframe.append(df)
        collectionDataframe = collectionDataframe.set_index("#obj")
        return collectionDataframe

    def getCollectionsListFromMasterfile(self):
        # get list of collection files from masterfile
        master_id = open(os.path.abspath(self.masterfile_dir + '/' + self.masterfile_name), 'r')
        collections_list = []
        for line, collection in enumerate(master_id):
            collections_list.append(collection.strip())
        master_id.close()
        return collections_list

    def getTrainIndex(self):
        """Returns index of one training sample that has not yet been used in the current epoch.
        """

        # Get 'random' index of sample from indices of unused samples
        indexTrain = self.indexListTraining[self.nextUnusedTrainingSampleIndex]
        self.nextUnusedTrainingSampleIndex += 1

        # Re-Shuffle all indices if all samples have been used
        if self.nextUnusedTrainingSampleIndex == self.amountOfTrainingSamples:
            self.nextUnusedTrainingSampleIndex = 0
            self.epochsCompletedTraining += 1
            random.shuffle(self.indexListTraining)
            print("Completed Training Epochs:", self.epochsCompletedTraining)

        return indexTrain

    def getValidIndex(self):
        """Returns index of one validation sample that has not yet been used in the current epoch.
        """

        # Get 'random' index of sample from indices of unused samples
        indexValid = self.indexListValidation[self.nextUnusedValidationSampleIndex]
        self.nextUnusedValidationSampleIndex += 1

        # Re-Shuffle all indices if all samples have been used
        if self.nextUnusedValidationSampleIndex == self.amountOfValidationSamples:
            self.nextUnusedValidationSampleIndex = 0
            self.epochsCompletedValidation += 1
            random.shuffle(self.indexListValidation)
        # else:
        #     print("no shuffle")

        return indexValid

    def load_image_data(self, collections_dict_MTL_train, image_2_label_dict_train,
                        collections_dict_MTL_validation, image_2_label_dict_validation):
        print("Loading all images...")
        self.data_all = {'train': None, 'valid': None}

        if self.sampleType == "image":
            self.data_all['train'] = self.load_image_data_image_based(collections_dict_MTL=collections_dict_MTL_train,
                                                                      image_2_label_dict=image_2_label_dict_train)

            self.data_all['valid'] = self.load_image_data_image_based(
                collections_dict_MTL=collections_dict_MTL_validation,
                image_2_label_dict=image_2_label_dict_validation)

        elif self.sampleType == "record":
            self.data_all['train'] = self.load_image_data_record_based(collections_dict_MTL=collections_dict_MTL_train,
                                                                       image_2_label_dict=image_2_label_dict_train)

            self.data_all['valid'] = self.load_image_data_record_based(
                collections_dict_MTL=collections_dict_MTL_validation,
                image_2_label_dict=image_2_label_dict_validation)

        elif self.sampleType == "mixed":
            self.data_all['train'] = self.load_image_text_data(collections_dict_MTL=collections_dict_MTL_train,
                                                               image_2_label_dict=image_2_label_dict_train)

            self.data_all['valid'] = self.load_image_text_data(collections_dict_MTL=collections_dict_MTL_validation,
                                                               image_2_label_dict=image_2_label_dict_validation)

    def load_image_data_image_based(self, collections_dict_MTL, image_2_label_dict):
        """Loads data for the image-based scenario.
            :Arguments:
              :collections_dict_MTL (*string*)\::
                  The collections_dict for the data that shall be loaded.
                  collections_dict_MTL: {task1: {label1: [listOfImages], label2: [listOfImages]}, task2:...}, 'nan' not included

              :image_2_label_dict (*string*)\::
                  The image_2_label_dict for the data that shall be loaded.
                  image_2_label_dict: {image1: {task1: label, task2: label}, image2: ...}

        """
        data_dict = {}
        for image_name in tqdm(list(image_2_label_dict.keys()), total=len(list(image_2_label_dict.keys()))):
            temp_ground_truth = []
            for MTL_label in collections_dict_MTL.keys():
                if MTL_label in list(image_2_label_dict[image_name].keys()):
                    label_name = image_2_label_dict[image_name][MTL_label][0]
                    if label_name == 'nan' or label_name == 'NaN':
                        class_label = -1
                    else:
                        class_label = list(collections_dict_MTL[MTL_label].keys()).index(
                            image_2_label_dict[image_name][MTL_label][0])
                else:
                    class_label = -1
                    label_name = 'NaN'
                temp_ground_truth.append(class_label)

            # load raw JPEG data from image path
            image_full_path = os.path.abspath(os.path.join(self.masterfile_dir, image_name))
            if not tf.io.gfile.exists(image_full_path):
                tf.logging.fatal('File does not exist %s', image_full_path)
            raw_image = tf.io.gfile.GFile(image_full_path, 'rb').read()

            if not self.multiLabelsListOfVariables is None:
                temp_ground_truth = self.convertGroundTruthToIndiceMatrix(temp_ground_truth)

            data_dict[image_name] = {'data': raw_image, 'labels': temp_ground_truth}

        return data_dict

    def load_image_data_record_based(self, collections_dict_MTL, image_2_label_dict):
        """Loads data for the record-based scenario.
                    :Arguments:
                      :collections_dict_MTL (*string*)\::
                          The collections_dict for the data that shall be loaded.
                          collections_dict_MTL: {task1: {label1: [listOfImages], label2: [listOfImages]}, task2:...}, 'nan' not included

                      :image_2_label_dict (*string*)\::
                          The image_2_label_dict for the data that shall be loaded.
                          image_2_label_dict: {image1: {task1: label, task2: label}, image2: ...}

        """
        data_dict = {}
        for object_name in tqdm(list(image_2_label_dict.keys()), total=len(list(image_2_label_dict.keys()))):
            temp_ground_truth = []
            for MTL_label in collections_dict_MTL.keys():
                if MTL_label in list(image_2_label_dict[object_name].keys()):
                    label_name = image_2_label_dict[object_name][MTL_label][0]
                    if label_name == 'nan' or label_name == 'NaN':
                        class_label = -1
                    else:
                        class_label = list(collections_dict_MTL[MTL_label].keys()).index(
                            image_2_label_dict[object_name][MTL_label][0])
                else:
                    class_label = -1
                    label_name = 'NaN'
                temp_ground_truth.append(class_label)

            # get list of images for this object
            imglist = self.collectionDataframe.loc[object_name.split("\\")[-1]]["#images"].split("#")[1:]
            imgdict = {}

            for imgname in imglist:
                # load raw JPEG data from image path
                image_full_path = os.path.abspath(os.path.join(self.masterfile_dir, imgname))
                if not tf.gfile.Exists(image_full_path):
                    tf.logging.fatal('File does not exist %s', image_full_path)
                raw_image = tf.gfile.GFile(image_full_path, 'rb').read()
                imgdict[imgname] = raw_image

            data_dict[object_name] = {'data': imgdict, 'labels': temp_ground_truth}

        return data_dict

    def load_image_text_data(self, collections_dict_MTL, image_2_label_dict):
        # read masterfile as csv to access lists of images for each object
        ds = self.collectionDataframe

        # Deskriptoren und Obj-IDs für Text-Samples laden
        file_descriptors = open(os.path.join(self.masterfile_dir, "text_training_vectors.txt").strip("\n"), 'r')
        file_obj = open(os.path.join(self.masterfile_dir, "text_training_ID.ids").strip("\n"), 'r')
        list_obj = [f.strip("\n") for f in file_obj]
        list_descriptors = [np.asarray(d.split(" "), dtype=np.float64) for d in file_descriptors]

        data_dict = {}
        for object_name in tqdm(list(image_2_label_dict.keys()), total=len(list(image_2_label_dict.keys()))):
            temp_ground_truth = []
            for MTL_label in collections_dict_MTL.keys():
                if MTL_label in list(image_2_label_dict[object_name].keys()):
                    label_name = image_2_label_dict[object_name][MTL_label][0]
                    if label_name == 'nan' or label_name == 'NaN':
                        class_label = -1
                    else:
                        class_label = list(collections_dict_MTL[MTL_label].keys()).index(
                            image_2_label_dict[object_name][MTL_label][0])
                else:
                    class_label = -1
                    label_name = 'NaN'
                temp_ground_truth.append(class_label)

            # get list of images for this object
            imglist = ds.loc[object_name.split("\\")[-1]]["#images"].split("#")[1:]
            imgdict = {}

            for imgname in imglist:
                # load raw JPEG data from image path
                image_full_path = os.path.abspath(os.path.join(self.masterfile_dir, imgname))
                if not tf.gfile.Exists(image_full_path):
                    tf.logging.fatal('File does not exist %s', image_full_path)
                raw_image = tf.gfile.GFile(image_full_path, 'rb').read()
                imgdict[imgname] = raw_image

            text_descriptor = list_descriptors[list_obj.index(object_name.split("\\")[-1])]

            data_dict[object_name] = {'data': imgdict, 'text_descriptor': text_descriptor, 'labels': temp_ground_truth}

        return data_dict

    def get_random_samples(self, how_many, purpose, session,
                           jpeg_data_tensor, decoded_image_tensor):
        if self.sampleType == "mixed":
            all_images, ground_truths, all_text_descriptors = self.get_random_samples_mixed(
                how_many, purpose, session, jpeg_data_tensor, decoded_image_tensor)
            return all_images, ground_truths, all_text_descriptors

        elif self.sampleType == "image":
            all_images, ground_truths, names = self.get_random_samples_image_based(
                how_many, purpose, session, jpeg_data_tensor, decoded_image_tensor)

        elif self.sampleType == "record":
            all_images, ground_truths, names = self.get_random_samples_record_based(
                how_many, purpose, session, jpeg_data_tensor, decoded_image_tensor)

        return all_images, ground_truths, names

    def get_random_samples_record_based(self, how_many, purpose, session,
                                        jpeg_data_tensor, decoded_image_tensor):
        if not self.multiLabelsListOfVariables is None:
            print("to be implemented")
            sys.exit()

        all_images = []
        ground_truths = []
        object_names = []

        # Retrieve a random sample of raw JPEG data
        for unused_i in range(how_many):
            # get index of unused sample
            image_index = None
            if purpose == 'train':
                image_index = self.getTrainIndex()
            if purpose == 'valid':
                image_index = self.getValidIndex()

            object_name = list(self.data_all[purpose].keys())[image_index]
            temp_ground_truth = self.data_all[purpose][object_name]['labels']

            # randomly choose one of the available images for the chosen object
            chosen_image = random.choice(list(self.data_all[purpose][object_name]['data'].values()))

            image_data = session.run(decoded_image_tensor,
                                     {jpeg_data_tensor: chosen_image})

            object_names.append(object_name)
            all_images.append(image_data)
            ground_truths.append(temp_ground_truth)

        ground_truths = tuple(zip(*ground_truths))
        return all_images, ground_truths, object_names

    def get_random_samples_image_based(self, how_many, purpose, session,
                                       jpeg_data_tensor, decoded_image_tensor):

        all_images = []
        ground_truths = []
        filenames = []
        all_image_index_train = []
        all_image_index_validation = []

        # Retrieve a random sample of raw JPEG data
        for unused_i in range(how_many):
            # get index of unused sample
            image_index = None
            if purpose == 'train':
                image_index = self.getTrainIndex()
                if self.amountOfTrainingSamples > how_many:
                    while image_index in all_image_index_train:
                        image_index = self.getTrainIndex()
                    all_image_index_train.append(image_index)
                else:
                    assert self.amountOfTrainingSamples > how_many, "select a smaller train batch size"
            if purpose == 'valid':
                image_index = self.getValidIndex()
                if self.amountOfValidationSamples>how_many:
                    while image_index in all_image_index_validation:
                        image_index = self.getValidIndex()
                    all_image_index_validation.append(image_index)
                else:
                    assert self.amountOfValidationSamples > how_many, "select a smaller batch size"

            image_name = list(self.data_all[purpose].keys())[image_index]
            temp_ground_truth = self.data_all[purpose][image_name][
                'labels']  # order of labels acc keys in self.taskDict
            image_data = session.run(decoded_image_tensor,
                                     {jpeg_data_tensor: self.data_all[purpose][image_name]['data']})
            all_images.append(image_data)
            ground_truths.append(temp_ground_truth)
            filenames.append(image_name)
        # print(np.shape(ground_truths))
        if self.multiLabelsListOfVariables is None:
            ground_truths = tuple(zip(*ground_truths))
        # arrange gt for multi-label classification here
        if self.boolMultiLabelClassification:
            ground_truths = tuple(zip(*ground_truths))

        # print(len(all_image_index_validation), how_many)
        return all_images, ground_truths, filenames

    def get_random_samples_mixed(self, how_many, purpose, session,
                                 jpeg_data_tensor, decoded_image_tensor):

        if not self.multiLabelsListOfVariables is None:
            print("to be implemented")
            sys.exit()

        all_images = []
        ground_truths = []
        all_text_descriptors = []

        # Retrieve a random sample of raw JPEG data
        for unused_i in range(how_many):
            # get index of unused sample
            image_index = None
            if purpose == 'train':
                image_index = self.getTrainIndex()
            if purpose == 'valid':
                image_index = self.getValidIndex()

            object_name = list(self.data_all[purpose].keys())[image_index]
            temp_ground_truth = self.data_all[purpose][object_name]['labels']

            # randomly choose one of the available images for the chosen object
            chosen_image = random.choice(list(self.data_all[purpose][object_name]['data'].values()))

            image_data = session.run(decoded_image_tensor,
                                     {jpeg_data_tensor: chosen_image})

            text_descriptor = self.data_all[purpose][object_name]['text_descriptor']

            all_images.append(image_data)
            ground_truths.append(temp_ground_truth)
            all_text_descriptors.append(text_descriptor)

        ground_truths = tuple(zip(*ground_truths))
        return all_images, ground_truths, all_text_descriptors

    def convertGroundTruthToIndiceMatrix(self, temp_ground_truth):
        converted_ground_truth = []
        for task_ind, task in enumerate(self.taskDict.keys()):
            if task in self.multiLabelsListOfVariables:
                if self.boolMultiLabelClassification:
                    list_of_multilabel_ind = [ind for ind, name_str in enumerate(self.taskDict[task]) if "___" in name_str]
                    if temp_ground_truth[task_ind] in list_of_multilabel_ind:
                        converted_label = np.zeros(len(self.classPerTaskDict[task]))
                        for single_label in self.taskDict[task][temp_ground_truth[task_ind]].split("___"):
                            temp_unconverted = list(self.classPerTaskDict[task]).index(single_label)
                            temp_converted = self.get_one_hot_label(indice=temp_unconverted,
                                                                    depth=len(self.classPerTaskDict[task]))
                            converted_label = converted_label + temp_converted
                    else:
                        if temp_ground_truth[task_ind] < 0:
                            temp_unconverted = temp_ground_truth[task_ind]
                        else:
                            temp_unconverted = list(self.classPerTaskDict[task]).index(
                                self.taskDict[task][temp_ground_truth[task_ind]])
                        converted_label = self.get_one_hot_label(indice=temp_unconverted,
                                                                 depth=len(self.classPerTaskDict[task]))
                    converted_ground_truth.append(list(converted_label))
                else:
                    list_of_multilabel_ind = [ind for ind, name_str in enumerate(self.taskDict[task]) if "___" in name_str]
                    if temp_ground_truth[task_ind] in list_of_multilabel_ind:
                        converted_label = np.zeros(self.max_num_classes_per_variable)
                        for single_label in self.taskDict[task][temp_ground_truth[task_ind]].split("___"):
                            temp_unconverted = list(self.taskDict[task]).index(single_label)
                            temp_converted = self.get_one_hot_label(indice=temp_unconverted,
                                                                    depth=self.max_num_classes_per_variable)
                            converted_label = converted_label + temp_converted
                    else:
                        converted_label = self.get_one_hot_label(indice=temp_ground_truth[task_ind],
                                                                 depth=self.max_num_classes_per_variable)
                    converted_ground_truth.append(list(converted_label))
            else:
                if self.boolMultiLabelClassification:
                    if temp_ground_truth[task_ind] < 0:
                        temp_unconverted = temp_ground_truth[task_ind]
                    else:
                        temp_unconverted = list(self.classPerTaskDict[task]).index(
                            self.taskDict[task][temp_ground_truth[task_ind]])
                    # print(temp_unconverted)
                    converted_label = self.get_one_hot_label(indice=temp_unconverted,
                                                             depth=len(self.classPerTaskDict[task]))
                else:
                    converted_label = self.get_one_hot_label(indice=temp_ground_truth[task_ind],
                                                             depth=self.max_num_classes_per_variable)
                converted_ground_truth.append(converted_label)
        return converted_ground_truth

    def get_one_hot_label(self, indice, depth):
        converted = list(np.zeros(depth))
        if indice >= 0:
            converted[indice] = 1
        return converted

    def splitDataIntoTrainingAndValidation(self, collections_dict_MTL, image_2_label_dict, validation_percentage):
        """Splits the whole collection into two collections.

        :Arguments:
            :collections_dict_MTL (*dictionary*)\::
                It's a dict of dicts. The keys are the tasks, the values are again
                dictionaries. The keys of those dictionaries are the class labels of
                the aforementioned task, the values are lists of images.
                collections_dict_MTL[task][class label][images]
            :image_2_label_dict (*dictionary*)\::
                A dictionary with the image (base) name as key and a list of avaiable
                #labels as value. It's needed for the estimation of the multitask
                loss.
                image_2_label_dict[full_image_name][#label_1, ..., #label_N]
            :validation_percentage (*int*)\::
                Percentage that is used for validation. For example, a value of 20
                means that 80% of the images in image_2_label_dict will be used for
                training, the other 20% will be used for validation.

        :Returns:
            :collections_dict_MTL_train (*dictionary*)\::
                It's a dict of dicts. The keys are the tasks, the values are again
                dictionaries. The keys of those dictionaries are the class labels of
                the aforementioned task, the values are lists of images used for training.
                collections_dict_MTL[task][class label][images]
            :collections_dict_MTL_val (*dictionary*)\::
                It's a dict of dicts. The keys are the tasks, the values are again
                dictionaries. The keys of those dictionaries are the class labels of
                the aforementioned task, the values are lists of images used for validation.
                collections_dict_MTL[task][class label][images]
            :image_2_label_dict_train (*dictionary*)\::
                A dictionary with the image (base) name as key and a list of avaiable
                #labels as value. It's needed for the estimation of the multitask
                loss. The images are used for training.
                image_2_label_dict_train[full_image_name][#label_1, ..., #label_N]
            :image_2_label_dict_val (*dictionary*)\::
                A dictionary with the image (base) name as key and a list of avaiable
                #labels as value. It's needed for the estimation of the multitask
                loss. The images are used for validation.
                image_2_label_dict_val[full_image_name][#label_1, ..., #label_N]

        """

        if validation_percentage == 0:
            collections_dict_MTL_train = collections_dict_MTL
            collections_dict_MTL_val = {}
            image_2_label_dict_train = image_2_label_dict
            image_2_label_dict_val = {}
            return collections_dict_MTL_train, collections_dict_MTL_val, image_2_label_dict_train, image_2_label_dict_val

        collections_dict_MTL_train = {}
        collections_dict_MTL_val = {}
        image_2_label_dict_train = {}
        image_2_label_dict_val = {}
        num_train = int(np.floor(len(list(image_2_label_dict.keys())) *
                                 (100 - validation_percentage) / 100))
        #    num_val   = len(list(image_2_label_dict.keys())) - num_train

        # Randomly choose samples for training and validation
        idx_all = np.asarray(range(len(list(image_2_label_dict.keys()))))
        random.seed(5517)
        random.shuffle(idx_all)
        idx_train = idx_all[:num_train]
        idx_valid = idx_all[num_train:]
        for image_index in idx_train:
            image_key = list(image_2_label_dict.keys())[image_index]
            image_2_label_dict_train[image_key] = image_2_label_dict[image_key]
        for image_index in idx_valid:
            image_key = list(image_2_label_dict.keys())[image_index]
            image_2_label_dict_val[image_key] = image_2_label_dict[image_key]

        # Iterate over tasks (timespan/place/...)
        for MTL_key in collections_dict_MTL.keys():
            # Iterate over related class labels
            for class_key in collections_dict_MTL[MTL_key].keys():
                # all images with that class label
                for image in collections_dict_MTL[MTL_key][class_key]:
                    fullname_image = os.path.abspath(os.path.join(self.masterfile_dir,
                                                                  image))
                    # Part of Training
                    if fullname_image in image_2_label_dict_train.keys():
                        if MTL_key not in collections_dict_MTL_train.keys():
                            temp_dict = {class_key: [fullname_image]}
                            collections_dict_MTL_train[MTL_key] = temp_dict
                        else:
                            if class_key not in collections_dict_MTL_train[MTL_key].keys():
                                collections_dict_MTL_train[MTL_key][class_key] = [fullname_image]
                            else:
                                collections_dict_MTL_train[MTL_key][class_key].append(fullname_image)
                    # Part of Validation
                    else:
                        if MTL_key not in collections_dict_MTL_val.keys():
                            temp_dict = {class_key: [fullname_image]}
                            collections_dict_MTL_val[MTL_key] = temp_dict
                        else:
                            if class_key not in collections_dict_MTL_val[MTL_key].keys():
                                collections_dict_MTL_val[MTL_key][class_key] = [fullname_image]
                            else:
                                collections_dict_MTL_val[MTL_key][class_key].append(fullname_image)
        #    print('An equal class distribution in training and validation is not',
        #          'guaranteed by now.')
        return (collections_dict_MTL_train, collections_dict_MTL_val,
                image_2_label_dict_train, image_2_label_dict_val)

    def getTaskDictFromCollectionDict(self, collections_dict_MTL):
        taskDict = {}
        for task in collections_dict_MTL:
            taskDict[task] = list(collections_dict_MTL[task].keys())
        return taskDict

    def printNumberOfSamplesForEveryClass(self, collections_dict_MTL_train, collections_dict_MTL_val):
        print("***************************************\n"
              "** Number of samples for every class **\n"
              "***************************************\n")
        for MTL_task in collections_dict_MTL_train.keys():
            print("%30s %5s %5s %5s" % ("***" + MTL_task.upper() + "***", "Train", "Valid", "All"))
            for class_ in collections_dict_MTL_train[MTL_task]:
                amountTrain = len(collections_dict_MTL_train[MTL_task][class_])
                if self.validation_percentage > 0:
                    amountValid = len(collections_dict_MTL_val[MTL_task][class_])
                    print("%30s: %5i %5i %5i" % (class_, amountTrain, amountValid, amountTrain + amountValid))
                else:
                    amountValid = 0
                    print("%30s: %5i %5i %5i" % (class_, amountTrain, amountValid, amountTrain + amountValid))
            print("")
