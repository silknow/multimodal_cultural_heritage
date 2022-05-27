import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import sys
import tensorflow_hub as hub
from tqdm import tqdm
import urllib
import pandas as pd
import urllib.request
import math

try:
    sys.path.insert(0, '../')
    import SILKNOW_WP4_library as wp4lib
    from SampleHandler import SampleHandler
    import LossCollections
except:
    from . import SILKNOW_WP4_library as wp4lib
    from . import SampleHandler
    from . import LossCollections

class ImportGraph():
    """  Importing and running isolated TF graph """

    def __init__(self, loc):
        # Create local graph and use it in the session
        self.multi_label_variables = None
        self.sess = tf.compat.v1.Session()
        signature_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        input_key = 'x_input'
        input_key_2 = 'x2_input'

        meta_graph_def = tf.saved_model.loader.load(
            self.sess,
            [tf.saved_model.SERVING],
            os.path.join(loc, "trained_model"))
        signature = meta_graph_def.signature_def

        self.task_list = []
        self.classification_tensors = []
        for key in list(signature[signature_key].outputs.keys()):
            if key != "y_output":
                self.classification_tensors.append(signature[signature_key].outputs[key].name)
                self.task_list.append(key.split("_")[-1])
        self.class_in_tensors = [signature[signature_key].inputs[key].name for key in
                                 list(signature[signature_key].inputs.keys()) if "mtl" in key]

        input_tensor_name = signature[signature_key].inputs[input_key].name
        input2_tensor_name = signature[signature_key].inputs[input_key_2].name
        self.raw_input_tensor = self.sess.graph.get_tensor_by_name(input_tensor_name)
        self.input_tensor = self.sess.graph.get_tensor_by_name(input2_tensor_name)

    def run(self, data):
        """ Running the activation operation previously imported.

        :Arguments:
            :data:
                The image data, i.e. the output from read_tensor_from_image_file.

        :Returns:
            :output:
                The result of the specified layer (output_name).
        """
        # The 'x' corresponds to name of input placeholder
        feed_dict_raw = {self.raw_input_tensor: data}
        decoded_img_op = self.sess.graph.get_operation_by_name('JPG-Decoding/Squeeze').outputs[0]
        decoded_img = self.sess.run(decoded_img_op, feed_dict=feed_dict_raw)
        decoded_img = np.expand_dims(np.asarray(decoded_img), 0)
        feed_dict_decoded = {self.input_tensor: decoded_img}

        for gt_tensor, task in zip(self.class_in_tensors, self.task_list):
            try:  # multi-class
                feed_dict_decoded[gt_tensor] = [0.]
            except:  # mixed multi-class and multi-label
                try:  # multi-class (softmax)
                    feed_dict_decoded[gt_tensor] = [[0.] * gt_tensor.shape[-1]]
                except:  # mutli-label (sigmoid)
                    if self.multi_label_variables is None:
                        self.multi_label_variables = [task]
                    else:
                        self.multi_label_variables.append(task)
                    feed_dict_decoded[gt_tensor] = [[0.] * gt_tensor.shape[-1]]
        activations = self.sess.run(self.classification_tensors, feed_dict=feed_dict_decoded)

        return activations


class SilkClassifier:
    """Class for handling all functions of the classifier."""

    def __init__(self):
        """Creates an empty object of class silk_classifier."""
        # record-based samples or image-based samples?
        self.image_based_samples = None
        self.bool_unlabeled_dataset = None

        # Directories
        self.masterfile_name = None
        self.masterfile_name_cv = None
        self.masterfile_dir = None
        self.log_dir = None
        self.log_dir_cv = None
        self.result_dir = None
        self.model_dir = None
        self.pred_gt_dir = None

        # Network Architecture
        self.num_joint_fc_layer = None
        self.num_nodes_joint_fc = None
        self.num_finetune_layers = None
        self.final_tensor_name = 'final_result'
        self.add_fc=[]

        # Training Specifications
        self.relevant_variables = None
        self.tfhub_module = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1"
        self.batchsize = None
        self.how_many_training_steps = None
        self.how_often_validation = None
        self.validation_percentage = None
        self.learning_rate = None
        self.weight_decay = None
        self.num_task_stop_gradient = None
        self.dropout_rate = None
        self.nameOfLossFunction = None
        self.lossParameters = None
        self.multiLabelsListOfVariables = None

        self.sigmoid_activation_thresh = 0.5

        self.restriction_Anne = False

        # Augmentation
        self.aug_set_dict = {}

    """MAIN FUNCTIONS"""

    def train_model(self):
        """Trains a new classifier.

            :Arguments\::
                No arguments. All parameters have to be set within the class object.

            :Returns\::
                No returns. The trained graph (containing the tfhub_module and the
                trained classifier) is stored automatically in the directory given in
                the control file.
            """

        # Assertions and Paths
        assert 0 <= self.validation_percentage <= 100, "Validation Percentage has to be between 0 and 100!"
        if (self.multiLabelsListOfVariables is None and self.nameOfLossFunction == "mixed_sce") \
                or (not self.multiLabelsListOfVariables is None and self.nameOfLossFunction != "mixed_sce"):
            print("Either pass multi-label variables or set the loss to mixed_sce. Multi_label variables has to be None for all nameOfLossFunction != mixed_sce.")
            sys.exit(-1)

        if not os.path.exists(os.path.join(self.log_dir, r"")): os.makedirs(os.path.join(self.log_dir, r""))

        self._write_train_parameters_to_configuration_file()

        # Initialize the SampleHandler
        boolMultiLabelClassification = isinstance(self.multiLabelsListOfVariables, list)
        self.samplehandler = SampleHandler(masterfile_dir=self.masterfile_dir,
                                           masterfile_name=self.masterfile_name,
                                           relevant_variables=self.relevant_variables,
                                           image_based_samples=self.image_based_samples,
                                           validation_percentage=self.validation_percentage,
                                           multiLabelsListOfVariables=self.multiLabelsListOfVariables,
                                           boolMultiLabelClassification=boolMultiLabelClassification)

        # Setup graph
        module_spec = hub.load_module_spec(str(self.tfhub_module))
        with tf.Graph().as_default() as graph:
            input_image_tensor, \
            bottleneck_tensor = self.create_module_graph(module_spec=module_spec)

            ground_truth_input, \
            final_tensor, \
            logits_MTL, \
            dropout_rate_input_tensor = self.add_MTL_graph(bottleneck_tensor)

            with tf.name_scope("loss-function"):
                lossFunction = LossCollections.getLossFunction(self.nameOfLossFunction,
                                                               self.lossParameters)
                if boolMultiLabelClassification:
                    boolSigmoidActivation = {}
                    for task in self.samplehandler.taskDict.keys():
                        if task in self.multiLabelsListOfVariables:
                            boolSigmoidActivation[task] = True
                        else:
                            boolSigmoidActivation[task] = False
                    lossTensor = lossFunction(ground_truth_input, logits_MTL,
                                              boolSigmoidActivation)
                else:
                    lossTensor = lossFunction(ground_truth_input, logits_MTL,
                                              self.samplehandler.classCountDict)

            with tf.name_scope("optimizer"):
                train_step, \
                cross_entropy = self.addOptimizer(loss=lossTensor)

        # Initialize Session
        with tf.Session(graph=graph) as sess:
            # Initialize all weights: for the module to their pretrained values,
            # and for the newly added retraining layer to random initial values.
            init = tf.global_variables_initializer()
            sess.run(init)

            # Add JPEG-Decoding and data augmentation
            with tf.name_scope('JPG-Decoding'):
                jpeg_data_tensor, decoded_image_tensor = wp4lib.add_jpeg_decoding(module_spec=module_spec)
            with tf.name_scope('Data-Augmentation'):
                augmented_image_tensor = wp4lib.add_data_augmentation(self.aug_set_dict, decoded_image_tensor)

            # Merge all the summaries and write them out to the logpath
            merged = tf.compat.v1.summary.merge_all()
            train_writer = tf.compat.v1.summary.FileWriter(self.log_dir + '/train', sess.graph)

            # Create a train saver that is used to restore values into an eval graph
            # when exporting models.
            train_saver = tf.compat.v1.train.Saver()

            # after all bottlenecks are cached, the collections_dict_train will
            # be renamed to collections_dict, if a validation is desired
            if self.validation_percentage > 0:
                val_writer = tf.compat.v1.summary.FileWriter(self.log_dir + '/val', sess.graph)

            # Start training iterations
            best_validation_loss = -1
            for i in range(self.how_many_training_steps):
                (image_data,
                 train_ground_truth, train_image_name) = self.samplehandler.get_random_samples(how_many=self.batchsize,
                                                                                               purpose='train',
                                                                                               session=sess,
                                                                                               jpeg_data_tensor=jpeg_data_tensor,
                                                                                               decoded_image_tensor=decoded_image_tensor)

                # print(train_ground_truth)

                # Online Data Augmentation
                vardata = [sess.run(augmented_image_tensor, feed_dict={decoded_image_tensor: imdata}) for imdata in
                           image_data]
                image_data = vardata

                feed_dictionary = {input_image_tensor: image_data, dropout_rate_input_tensor: self.dropout_rate}
                for ind in range(np.shape(train_ground_truth)[0]):
                    feed_dictionary[ground_truth_input[ind]] = train_ground_truth[ind]

                (train_summary,
                 cross_entropy_value, _) = sess.run(
                    [merged, cross_entropy, train_step],
                    feed_dict=feed_dictionary)
                train_writer.add_summary(train_summary, i)
                summary_train_loss = [tf.Summary.Value(tag='Losses', simple_value=cross_entropy_value)]
                train_writer.add_summary(tf.Summary(value=summary_train_loss), i)
                train_writer.flush()
                print("Training Loss in iteration %i: %2.6f" % (i, cross_entropy_value))

                # Optional Validation
                if self.validation_percentage > 0 and (i % self.how_often_validation == 0):
                    total_validation_loss = []
                    maximumNumberOfValidationIterations = int(
                        np.ceil(self.samplehandler.amountOfValidationSamples / self.batchsize))
                    for val_iter in range(maximumNumberOfValidationIterations):
                        # var_batch_size = min(self.batchsize,
                        #                      self.samplehandler.amountOfValidationSamples - self.samplehandler.nextUnusedValidationSampleIndex)
                        var_batch_size = self.batchsize

                        (val_image_data,
                         val_ground_truth, _) = self.samplehandler.get_random_samples(how_many=var_batch_size,
                                                                                      purpose='valid',
                                                                                      session=sess,
                                                                                      jpeg_data_tensor=jpeg_data_tensor,
                                                                                      decoded_image_tensor=decoded_image_tensor)

                        feed_dictionary = {input_image_tensor: val_image_data}
                        if self.multiLabelsListOfVariables is None:
                            val_ground_truth_ = np.asarray(val_ground_truth)
                        else:
                            val_ground_truth_ = list(val_ground_truth)
                        # print(val_ground_truth_)
                        # sys.exit()
                        # val_ground_truth_ = val_ground_truth

                        if (len(feed_dictionary[input_image_tensor])) == 0: break
                        for ind in range(np.shape(val_ground_truth_)[0]):
                            feed_dictionary[ground_truth_input[ind]] = val_ground_truth_[ind]

                        (validation_summary, validation_loss) = sess.run([merged, cross_entropy],
                                                                         feed_dict=feed_dictionary)
                        total_validation_loss.append(validation_loss)
                        # print(val_iter, maximumNumberOfValidationIterations, validation_loss)
                        # print("It took %2.2f seconds to process a validation batch" % ((dt.datetime.now() - t1).seconds + (dt.datetime.now() - t1).microseconds/1e6))

                    total_validation_loss = np.mean(total_validation_loss)
                    summary_val_loss = [tf.Summary.Value(tag='Losses', simple_value=total_validation_loss)]
                    val_writer.add_summary(tf.Summary(value=summary_val_loss), i)
                    val_writer.flush()
                    # print("Validation Loss: " + str(total_validation_loss))
                    print("Validation Loss in iteration %i: %2.6f" % (i, total_validation_loss))

                # Save (best) configuration
                if self.validation_percentage > 0:
                    if best_validation_loss > total_validation_loss or best_validation_loss == -1:
                        print("New best model found!")
                        train_saver.save(sess, self.log_dir + '/' + 'model.ckpt')
                        best_validation_loss = total_validation_loss
                # else:
                #     train_saver.save(sess, self.log_dir + '/' + 'model.ckpt')

            # Save the last model in case of no validation
            if self.validation_percentage == 0:
                train_saver.save(sess, self.log_dir + '/' + 'model.ckpt')
        # Save task_dict for subsequent functions
        np.savez(self.log_dir + r"/task_dict.npz", self.samplehandler.classPerTaskDict)
        self.save_final_model()

    def save_final_model(self):
        save_graph = tf.Graph()
        save_sess = tf.compat.v1.Session(graph=save_graph)
        with save_graph.as_default():
            # load best performing weights
            saver = tf.compat.v1.train.import_meta_graph(self.log_dir + 'model.ckpt' + '.meta',
                                                         clear_devices=True)
            init = tf.compat.v1.global_variables_initializer()
            save_sess.run(init)
            saver.restore(save_sess, self.log_dir + 'model.ckpt')

            # prepare inputs
            x = save_sess.graph.get_tensor_by_name("JPG-Decoding/DecodeJPGInput:0")
            x2 = save_sess.graph.get_tensor_by_name("moduleLayers/input_img:0")
            tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
            tensor_info_x2 = tf.saved_model.utils.build_tensor_info(x2)
            inputs = {'x_input': tensor_info_x,
                           'x2_input': tensor_info_x2}
            task_list = list(self.samplehandler.classPerTaskDict.keys())
            for task in task_list:
                try:  # multi-class
                    cur_in_tensor = save_sess.graph.get_tensor_by_name(
                        'customLayers/input/GroundTruthInput' + task + ":0")
                except:  # mixed multi-class and multi-label
                    try:  # multi-class (softmax)
                        cur_in_tensor = save_sess.graph.get_tensor_by_name(
                            'customLayers/input/GroundTruthInputMultiClass' + task + ":0")
                    except:  # mutli-label (sigmoid)
                        cur_in_tensor = save_sess.graph.get_tensor_by_name(
                            'customLayers/input/GroundTruthInputMultiLabel' + task + ":0")
                cur_in_tensor_info = tf.saved_model.utils.build_tensor_info(cur_in_tensor)
                inputs["input_mtl_" + task] = cur_in_tensor_info

            # prepare outputs
            outputs = {}
            for task in task_list:
                cur_tensor = save_sess.graph.get_operation_by_name(
                    'customLayers/' + self.final_tensor_name + '_' + task).outputs[0]
                cur_tensor_info = tf.saved_model.utils.build_tensor_info(cur_tensor)
                outputs["output_mtl_" + task] = cur_tensor_info

            # prepare model_dir
            self.model_dir= os.path.join(self.log_dir, "trained_model")
            if os.path.exists(self.model_dir):
                dir = self.model_dir
                for f in os.listdir(dir):
                    os.remove(os.path.join(dir, f))
                    if os.path.isdir(os.path.join(dir, f)):
                        for ff in os.path.join(dir, f):
                            os.remove(os.path.join(os.path.join(dir, f), ff))
            # try:
            #     os.removedirs(self.model_dir)
            # except:
            #     os.rmdir(self.model_dir)
            # if not os.exists(self.model_dir):
            #     os.mkdir(self.model_dir)

            # save model
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.model_dir)
            prediction_signature = (
                tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                    inputs=inputs,
                    outputs=outputs,
                    method_name=tf.saved_model.PREDICT_METHOD_NAME))
            builder.add_meta_graph_and_variables(
                save_sess, [tf.saved_model.SERVING],
                signature_def_map={
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        prediction_signature
                },
            )
            builder.save()

            np.savez(self.log_dir + r"/task_dict.npz", self.samplehandler.classPerTaskDict)

    def classify_images(self):
        """Uses a trained classifier for prediction.

            :Arguments\::
                :Lorem (*Ipsum*)\::
                    Dolor sit amet.

            :Returns\::
            """

        # Checks for paths existing: treepath, master file, savepath, model_path
        if not os.path.exists(os.path.join(self.result_dir, r"")): os.makedirs(os.path.join(self.result_dir, r""))

        # Load pre-trained network and task_dict
        task_dict = np.load(self.model_dir + r"/task_dict.npz", allow_pickle=True)["arr_0"].item()
        model = model = ImportGraph(self.model_dir)
        task_list = task_dict.keys()

        # Try to read master_file with known labels
        groundtruth = []
        try:
            coll_list = wp4lib.master_file_to_collections_list(self.masterfile_dir, self.masterfile_name)
            coll_dict, data_dict = wp4lib.collections_list_MTL_to_image_lists(
                collections_list=coll_list,
                labels_2_learn=task_list,
                master_dir=self.masterfile_dir,
                multiLabelsListOfVariables=self.multiLabelsListOfVariables,
                bool_unlabeled_dataset=self.bool_unlabeled_dataset)

            for dd in data_dict.keys():
                vardict = data_dict[dd]
                varlist = [vardict[task] for task in task_list]
                groundtruth.append(varlist)

            image_file_array = data_dict.keys()

            # read all collection files into one dataframe
            collectionDataframe = pd.DataFrame()
            if not self.image_based_samples:
                for cFile in coll_list:
                    df = pd.read_csv(os.path.join(self.masterfile_dir, cFile), delimiter="\t")
                    collectionDataframe = collectionDataframe.append(df)
                collectionDataframe = collectionDataframe.set_index("#obj")

        # read list of custom images that where not exported from the KG
        except:
            # Read list of images that are to be classified
            master_id = open(os.path.abspath(self.masterfile_dir + '/' + self.masterfile_name), 'r')
            collections_list = []
            for line, collection in enumerate(master_id):
                if "#" in collection and line == 0:
                    collections_list.append(self.masterfile_dir + '/' + self.masterfile_name)
                    break
                else:
                    collections_list.append(collection.strip())
            master_id.close()
            (image_file_array) = self.image_lists_to_image_array(collections_list)
            groundtruth = None

        # Prepare result files
        if "place" in task_list:
            sys_integration_pred_place = os.path.join(self.result_dir, "sys_integration_pred_place.csv")
            integtration_file_place = open(os.path.abspath(sys_integration_pred_place), 'w')
            integtration_file_place.write("obj_uri, museum, image_name, predicted_class, class_score\n")
        if "timespan" in task_list:
            sys_integration_pred_timespan = os.path.join(self.result_dir, "sys_integration_pred_timespan.csv")
            integtration_file_timespan = open(os.path.abspath(sys_integration_pred_timespan), 'w')
            integtration_file_timespan.write("obj_uri, museum, image_name, predicted_class, class_score\n")
        if "technique" in task_list:
            sys_integration_pred_technique = os.path.join(self.result_dir, "sys_integration_pred_technique.csv")
            integtration_file_technique = open(os.path.abspath(sys_integration_pred_technique), 'w')
            integtration_file_technique.write("obj_uri, museum, image_name, predicted_class, class_score\n")
        if "material" in task_list:
            sys_integration_pred_material = os.path.join(self.result_dir, "sys_integration_pred_material.csv")
            integtration_file_material = open(os.path.abspath(sys_integration_pred_material), 'w')
            integtration_file_material.write("obj_uri, museum, image_name, predicted_class, class_score\n")
        if "depiction" in task_list:
            sys_integration_pred_depiction = os.path.join(self.result_dir, "sys_integration_pred_depiction.csv")
            integtration_file_depiction = open(os.path.abspath(sys_integration_pred_depiction), 'w')
            integtration_file_depiction.write("obj_uri, museum, image_name, predicted_class, class_score\n")

        classification_result = self.result_dir + r"/classification_results.txt"
        classification_scores = self.result_dir + r"/classification_scores.txt"
        class_res_id = open(os.path.abspath(classification_result), 'w')
        class_score_file = open(os.path.abspath(classification_scores), 'w')
        class_res_id.write('#image_file')
        for task in task_list:
            class_res_id.write('\t#' + task)
        class_res_id.write('\n')

        predictions = []

        # when image_based_samples is True, image_file_array lists individual image files
        # when image_based_samples is False, image_file_array lists object URIs
        # in this case, the URIs have to be dereferenced into image files
        for image_file in tqdm(image_file_array, total=len(image_file_array)):
            if self.image_based_samples:
                im_file_full_path = os.path.abspath(os.path.join(self.masterfile_dir,
                                                                 image_file))
                image_data = tf.io.gfile.GFile(im_file_full_path, 'rb').read()
                results = model.run(image_data)

            else:
                object_name = image_file
                imglist = collectionDataframe.loc[object_name.split("\\")[-1]]["#images"].split("#")[1:]

                results_for_objects_images = []
                for imgname in imglist:
                    # load raw JPEG data from image path
                    image_full_path = os.path.abspath(os.path.join(self.masterfile_dir, imgname))
                    image_data = tf.gfile.GFile(image_full_path, 'rb').read()

                    results = model.run(image_data)

                    results_for_objects_images.append(results)

                # average results from all objects images
                results = np.mean(results_for_objects_images, axis=0)

            # Get most probable class(es) per task
            resultClasses = {}
            resultClasses_list = []
            for i, task in enumerate(task_list):
                class_list = task_dict[task]
                if self.multiLabelsListOfVariables is None:  # only multi-class classifications (softmax)
                    resultClasses[task] = class_list[np.argmax(results[i])]
                    resultClasses_list.append(class_list[np.argmax(results[i])])
                else:  # mixed multi-class (softmax) and mutli-label (sigmoid)
                    if task in self.multiLabelsListOfVariables:  # sigmoid case
                        multi_pred = np.where(results[i][0] > self.sigmoid_activation_thresh)[0]
                        if len(multi_pred) > 0:  # at least one label predicted
                            class_name = ""
                            for pred in multi_pred:
                                class_name = class_name + class_list[pred] + "___"
                            class_name = class_name[0:-3]
                            resultClasses[task] = class_name
                            resultClasses_list.append(class_name)
                        else:  # no label predicted; sigmoid <= self.sigmoid_activation_thresh for all classes
                            resultClasses[task] = "nan_OR_" + class_list[np.argmax(results[i])]
                            resultClasses_list.append("nan_OR_" + class_list[np.argmax(results[i])])
                    else:  # softmax case
                        resultClasses[task] = class_list[np.argmax(results[i])]
                        resultClasses_list.append(class_list[np.argmax(results[i])])

            # Write class names to file
            class_res_id.write("%s" % image_file)
            for task in task_list:
                class_res_id.write("\t%s" % (resultClasses[task]))
            class_res_id.write("\n")

            # OUTPUT OF CLASS SCORES
            class_score_file.write("****" + image_file + "****" + "\n")
            for ti, task in enumerate(task_dict):
                if self.multiLabelsListOfVariables is None:
                    activation = " (softmax)"
                else:
                    if task in self.multiLabelsListOfVariables:
                        activation = " (sigmoid)"
                    else:
                        activation = " (softmax)"
                class_score_file.write(task + activation + ": \t \t")
                for ci, c in enumerate(task_dict[task]):
                    class_score_file.write(c + ": " + str(np.around(results[ti][0][ci] * 100, 2)) + "% \t \t")

                if len(resultClasses[task].split("___")) > 1:
                    pred_class = resultClasses[task]
                    max_score = ""
                    for multi_class in pred_class.split("___"):
                        score = results[ti][0][list(task_dict[task]).index(multi_class)] * 100
                        max_score = max_score + str(np.around(score, 2)) + "___"
                    max_score = max_score[0:-3]
                else:
                    pred_class = resultClasses[task]
                    if "nan" in pred_class:
                        score = 1. - max(results[ti][0])
                        max_score = str(np.around(score, 2)) + "_OR_" + str(np.around(1. - score, 2))
                    else:
                        score = results[ti][0][list(task_dict[task]).index(resultClasses[task])] * 100
                        max_score = str(np.around(score, 2))

                if task == "place":
                    self.update_integration_file(image_file, integtration_file_place, max_score, pred_class)
                if task == "timespan":
                    self.update_integration_file(image_file, integtration_file_timespan, max_score, pred_class)
                if task == "technique":
                    self.update_integration_file(image_file, integtration_file_technique, max_score, pred_class)
                if task == "material":
                    self.update_integration_file(image_file, integtration_file_material, max_score, pred_class)
                if task == "depiction":
                    self.update_integration_file(image_file, integtration_file_depiction, max_score, pred_class)

                class_score_file.write("\n")
            class_score_file.write("\n")
            class_score_file.write("\n")

            predictions.append(resultClasses_list)

        class_res_id.close()
        class_score_file.close()
        if "place" in task_list:
            integtration_file_place.close()
        if "timespan" in task_list:
            integtration_file_timespan.close()
        if "technique" in task_list:
            integtration_file_technique.close()
        if "material" in task_list:
            integtration_file_material.close()
        if "depiction" in task_list:
            integtration_file_depiction.close()

        # Save Prediction and Groundtruth as .npy for evaluation
        pred_gt = {"Groundtruth": np.asarray(groundtruth),
                   "Predictions": np.asarray(predictions),
                   "task_dict": task_dict}
        np.savez(self.result_dir + r"/pred_gt.npz", pred_gt)

    def update_integration_file(self, image_file, integtration_file, max_score, pred_class):
        integtration_file.write(
            "http://data.silknow.org/object/" + os.path.basename(image_file).split("__")[1] + ", ")
        integtration_file.write(os.path.basename(image_file).split("__")[0] + ", ")
        integtration_file.write(os.path.basename(image_file).split("__")[2] + ", ")
        integtration_file.write(pred_class + ", ")
        integtration_file.write(str(max_score) + "%\n")

    def evaluate_model(self):
        r""" Evaluates a pre-trained model.

            :Arguments\::
                :pred_gt_path (*string*)\::
                    Path (without filename) to a "pred_gt.npz" file that was produced by
                    the function get_KNN.
                :result_path (*string*)\::
                    Path to where the evaluation results will be saved.

            :Returns\::
            """

        if not os.path.exists(os.path.join(self.result_dir, r"")): os.makedirs(os.path.join(self.result_dir, r""))

        # Load predictions and groundtruth
        vardict = np.load(self.pred_gt_dir + r"/pred_gt.npz", allow_pickle=True)["arr_0"].item()
        predictions = np.asarray(vardict["Predictions"])
        groundtruth = np.asarray(vardict["Groundtruth"])
        task_dict = vardict["task_dict"]

        if len(predictions.shape) == 1: predictions = np.expand_dims(predictions, -1)
        if len(groundtruth.shape) == 1: groundtruth = np.expand_dims(groundtruth, -1)

        label2class_list = []
        for task in task_dict.keys():
            var = [task]
            var.extend(task_dict[task])
            label2class_list.append(var)

        for task_ind, classlist in enumerate(label2class_list):
            taskname = classlist[0]
            list_class_names = np.asarray(classlist[1:])

            # sort out nans
            gtvar = np.squeeze(groundtruth[:, task_ind])
            prvar = np.squeeze(predictions[:, task_ind])
            nan_mask = np.squeeze(gtvar != 'nan')

            gtvar = gtvar[nan_mask]
            prvar = prvar[nan_mask]

            # prepare predictions and groundtruth according to multi-class and multi-label classification
            if self.multiLabelsListOfVariables is None:
                ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in gtvar])
                prediction = np.squeeze([np.where(pr == list_class_names) for pr in prvar])
                if len(np.unique(ground_truth)) < len(list_class_names):
                    list_class_names = [name for name in list_class_names if
                                        list(list_class_names).index(name) in np.unique(ground_truth)]
                wp4lib.estimate_quality_measures(ground_truth=ground_truth,
                                                 prediction=prediction,
                                                 list_class_names=list(list_class_names),
                                                 prefix_plot=taskname,
                                                 res_folder_name=self.result_dir)
            else:
                if taskname in self.multiLabelsListOfVariables:
                    wp4lib.create_multi_label_rectangle_confusion_matrix(groundtruth=gtvar,
                                                                         predictions=prvar,
                                                                         taskname=taskname,
                                                                         task_dict=task_dict,
                                                                         result_dir=self.result_dir)
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
                                wp4lib.estimate_quality_measures(ground_truth=ground_truth,
                                                                 prediction=prediction,
                                                                 list_class_names=list(
                                                                     ["no_" + class_name, class_name]),
                                                                 prefix_plot=taskname + "_binary_" + class_name,
                                                                 res_folder_name=self.result_dir)
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
                            wp4lib.estimate_quality_measures(ground_truth=ground_truth_all,
                                                             prediction=prediction_all,
                                                             list_class_names=list(["no_" + class_name, class_name]),
                                                             prefix_plot=taskname + "_binary_" + class_name + "_all",
                                                             res_folder_name=self.result_dir)
                            pred_all_whole_var.append(prediction_all)
                            gt_all_whole_var.append(ground_truth_all)
                        else:
                            print("(binary all) ground truth and predictions do not contain the class: ", class_name)

                    wp4lib.estimate_quality_measures(ground_truth=np.hstack(gt_no_nan_whole_var),
                                                     prediction=np.hstack(pred_no_nan_whole_var),
                                                     list_class_names=list(["not_class", "class"]),
                                                     prefix_plot=taskname + "_binary_whole_var",
                                                     res_folder_name=self.result_dir)
                    wp4lib.estimate_quality_measures(ground_truth=np.hstack(gt_all_whole_var),
                                                     prediction=np.hstack(pred_all_whole_var),
                                                     list_class_names=list(["not_class", "class"]),
                                                     prefix_plot=taskname + "_binary_whole_var_all",
                                                     res_folder_name=self.result_dir)
                else:
                    ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in gtvar])
                    prediction = np.squeeze([np.where(pr == list_class_names) for pr in prvar])
                    wp4lib.estimate_quality_measures(ground_truth=ground_truth,
                                                     prediction=prediction,
                                                     list_class_names=list(list_class_names),
                                                     prefix_plot=taskname,
                                                     res_folder_name=self.result_dir)

        return groundtruth, predictions

    def crossvalidation(self):
        r""" BlABLABLA"""

        # make sure path exists
        if not os.path.exists(os.path.join(self.log_dir_cv, r"")): os.makedirs(os.path.join(self.log_dir_cv, r""))

        # load masterfile
        coll_list = wp4lib.master_file_to_collections_list(self.masterfile_dir, self.masterfile_name_cv)

        # averaging results from all cviters
        predictions = []
        groundtruth = []

        # FIVE cross validation iterations
        logpath_cv = None
        for cviter in range(5):

            # create intermediate masterfiles for sub-modules
            if str(self.log_dir_cv)[-1] == "/":
                tempExperimentName = str(self.log_dir_cv).split("/")[-2]
            else:
                tempExperimentName = str(self.log_dir_cv).split("/")[-1]
            temporaryTrainMasterfileName = "Masterfile_train_" + tempExperimentName + ".txt"
            temporaryTestMasterfileName = "Masterfile_test_" + tempExperimentName + ".txt"
            train_master = open(os.path.abspath(self.masterfile_dir + '/' + temporaryTrainMasterfileName), 'w')
            test_master = open(os.path.abspath(self.masterfile_dir + '/' + temporaryTestMasterfileName), 'w')
            train_coll = np.roll(coll_list, cviter)[:-1]
            test_coll = np.roll(coll_list, cviter)[-1]
            for c in train_coll:
                train_master.write("%s\n" % c)
            test_master.write(test_coll)
            train_master.close()
            test_master.close()

            # set sub-logpath
            logpath_cv = self.log_dir_cv + r"/cv" + str(cviter) + "/"
            if not os.path.exists(logpath_cv):
                os.makedirs(logpath_cv)

            # perform training
            self.masterfile_name = temporaryTrainMasterfileName
            self.log_dir = logpath_cv
            self.model_dir = logpath_cv
            self.train_model()

            # predictions
            self.masterfile_name = temporaryTestMasterfileName
            self.log_dir = logpath_cv
            self.model_dir = logpath_cv
            self.result_dir = logpath_cv
            self.classify_images()

            # evaluations
            self.pred_gt_dir = logpath_cv
            gtvar, prvar = self.evaluate_model()

            # concatenate predictions and groundtruths
            if len(predictions) == 0:
                predictions = prvar
                groundtruth = gtvar
            else:
                predictions = np.concatenate((predictions, prvar))
                groundtruth = np.concatenate((groundtruth, gtvar))

            # delete intermediate data
            os.remove(self.masterfile_dir + '/' + temporaryTrainMasterfileName)
            os.remove(self.masterfile_dir + '/' + temporaryTestMasterfileName)

        # estimate quality measures with all predictions and groundtruths
        vardict = np.load(logpath_cv + r"/pred_gt.npz", allow_pickle=True)["arr_0"].item()
        task_dict = vardict["task_dict"]

        label2class_list = []
        for task in task_dict.keys():
            var = [task]
            var.extend(task_dict[task])
            label2class_list.append(var)

        for task_ind, classlist in enumerate(label2class_list):
            taskname = classlist[0]
            list_class_names = np.asarray(classlist[1:])

            # sort out nans
            gtvar = np.squeeze(groundtruth[:, task_ind])
            prvar = np.squeeze(predictions[:, task_ind])
            nan_mask = np.squeeze(gtvar != 'nan')

            gtvar = gtvar[nan_mask]
            prvar = prvar[nan_mask]

            # prepare predictions and groundtruth according to multi-class and multi-label classification
            if self.multiLabelsListOfVariables is None:
                ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in gtvar])
                prediction = np.squeeze([np.where(pr == list_class_names) for pr in prvar])
                wp4lib.estimate_quality_measures(ground_truth=ground_truth,
                                                 prediction=prediction,
                                                 list_class_names=list(list_class_names),
                                                 prefix_plot=taskname,
                                                 res_folder_name=self.log_dir_cv)
            else:
                if taskname in self.multiLabelsListOfVariables:
                    wp4lib.create_multi_label_rectangle_confusion_matrix(groundtruth=gtvar,
                                                                         predictions=prvar,
                                                                         taskname=taskname,
                                                                         task_dict=task_dict,
                                                                         result_dir=self.log_dir_cv)
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
                                wp4lib.estimate_quality_measures(ground_truth=ground_truth,
                                                                 prediction=prediction,
                                                                 list_class_names=list(
                                                                     ["no_" + class_name, class_name]),
                                                                 prefix_plot=taskname + "_binary_" + class_name,
                                                                 res_folder_name=self.log_dir_cv)
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
                            wp4lib.estimate_quality_measures(ground_truth=ground_truth_all,
                                                             prediction=prediction_all,
                                                             list_class_names=list(["no_" + class_name, class_name]),
                                                             prefix_plot=taskname + "_binary_" + class_name + "_all",
                                                             res_folder_name=self.log_dir_cv)
                            pred_all_whole_var.append(prediction_all)
                            gt_all_whole_var.append(ground_truth_all)
                        else:
                            print("(binary all) ground truth and predictions do not contain the class: ", class_name)

                    wp4lib.estimate_quality_measures(ground_truth=np.hstack(gt_no_nan_whole_var),
                                                     prediction=np.hstack(pred_no_nan_whole_var),
                                                     list_class_names=list(["not_class", "class"]),
                                                     prefix_plot=taskname + "_binary_whole_var",
                                                     res_folder_name=self.log_dir_cv)
                    wp4lib.estimate_quality_measures(ground_truth=np.hstack(gt_all_whole_var),
                                                     prediction=np.hstack(pred_all_whole_var),
                                                     list_class_names=list(["not_class", "class"]),
                                                     prefix_plot=taskname + "_binary_whole_var_all",
                                                     res_folder_name=self.log_dir_cv)
                else:
                    ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in gtvar])
                    prediction = np.squeeze([np.where(pr == list_class_names) for pr in prvar])
                    wp4lib.estimate_quality_measures(ground_truth=ground_truth,
                                                     prediction=prediction,
                                                     list_class_names=list(list_class_names),
                                                     prefix_plot=taskname,
                                                     res_folder_name=self.log_dir_cv)

        # for task_ind, classlist in enumerate(label2class_list):
        #     taskname = classlist[0]
        #     list_class_names = np.asarray(classlist[1:])
        #
        #     # sort out nans
        #     gtvar = np.squeeze(groundtruth[:, task_ind])
        #     prvar = np.squeeze(predictions[:, task_ind])
        #     nan_mask = np.squeeze(gtvar != 'nan')
        #
        #     gtvar = gtvar[nan_mask]
        #     prvar = prvar[nan_mask]
        #
        #     ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in gtvar])
        #     prediction = np.squeeze([np.where(pr == list_class_names) for pr in prvar])
        #     wp4lib.estimate_quality_measures(ground_truth=ground_truth,
        #                                      prediction=prediction,
        #                                      list_class_names=list(list_class_names),
        #                                      prefix_plot=taskname,
        #                                      res_folder_name=self.log_dir_cv)

    def trainWithDomainCheck(self):
        """Trains a new classifier.

            :Arguments\::
                No arguments. All parameters have to be set within the class object.

            :Returns\::
                No returns. The trained graph (containing the tfhub_module and the
                trained classifier) is stored automatically in the directory given in
                the control file.
            """

        # Assertions and Paths
        assert 0 <= self.validation_percentage <= 100, "Validation Percentage has to be between 0 and 100!"

        # Initialize the SampleHandler
        self.samplehandler = SampleHandler(masterfile_dir=self.masterfile_dir,
                                           masterfile_name=self.masterfile_name,
                                           relevant_variables=self.relevant_variables,
                                           image_based_samples=self.image_based_samples,
                                           validation_percentage=self.validation_percentage,
                                           multiLabelsListOfVariables=self.multiLabelsListOfVariables)

        self.samplehandlerTarget = SampleHandler(masterfile_dir=self.masterfile_dir,
                                                 masterfile_name=self.masterfileTarget,
                                                 relevant_variables=self.relevant_variables,
                                                 image_based_samples=self.image_based_samples,
                                                 validation_percentage=100.,
                                                 multiLabelsListOfVariables=self.multiLabelsListOfVariables)

        # Setup graph
        module_spec = hub.load_module_spec(str(self.tfhub_module))
        with tf.Graph().as_default() as graph:
            input_image_tensor, \
            bottleneck_tensor = self.create_module_graph(module_spec=module_spec)

            ground_truth_input, \
            final_tensor, \
            logits_MTL, \
            dropout_rate_input_tensor = self.add_MTL_graph(bottleneck_tensor)

            with tf.name_scope("loss-function"):
                lossFunction = LossCollections.getLossFunction(self.nameOfLossFunction,
                                                               self.lossParameters)
                loss = lossFunction(ground_truth_input, logits_MTL,
                                    self.samplehandler.classCountDict)

            with tf.name_scope("optimizer"):
                train_step, \
                cross_entropy = self.addOptimizer(loss=loss)

        # Initialize Session
        with tf.Session(graph=graph) as sess:
            # Initialize all weights: for the module to their pretrained values,
            # and for the newly added retraining layer to random initial values.
            init = tf.global_variables_initializer()
            sess.run(init)

            # Add JPEG-Decoding and data augmentation
            with tf.name_scope('JPG-Decoding'):
                jpeg_data_tensor, decoded_image_tensor = wp4lib.add_jpeg_decoding(module_spec=module_spec)
            with tf.name_scope('Data-Augmentation'):
                augmented_image_tensor = wp4lib.add_data_augmentation(self.aug_set_dict, decoded_image_tensor)

            # Merge all the summaries and write them out to the logpath
            merged = tf.compat.v1.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)

            # Create a train saver that is used to restore values into an eval graph
            # when exporting models.
            train_saver = tf.compat.v1.train.Saver()

            # after all bottlenecks are cached, the collections_dict_train will
            # be renamed to collections_dict, if a validation is desired
            if self.validation_percentage > 0:
                val_writer = tf.summary.FileWriter(self.log_dir + '/val', sess.graph)
            target_writer = tf.summary.FileWriter(self.log_dir + '/target', sess.graph)

            # Start training iterations
            best_validation_loss = -1
            for i in range(self.how_many_training_steps):
                (image_data,
                 train_ground_truth, _) = self.samplehandler.get_random_samples(how_many=self.batchsize,
                                                                                purpose='train',
                                                                                session=sess,
                                                                                jpeg_data_tensor=jpeg_data_tensor,
                                                                                decoded_image_tensor=decoded_image_tensor)

                # Online Data Augmentation
                vardata = [sess.run(augmented_image_tensor, feed_dict={decoded_image_tensor: imdata}) for imdata in
                           image_data]
                image_data = vardata

                feed_dictionary = {input_image_tensor: image_data, dropout_rate_input_tensor: self.dropout_rate}
                for ind in range(np.shape(train_ground_truth)[0]):
                    feed_dictionary[ground_truth_input[ind]] = train_ground_truth[ind]

                (train_summary,
                 cross_entropy_value, _) = sess.run(
                    [merged, cross_entropy, train_step],
                    feed_dict=feed_dictionary)
                train_writer.add_summary(train_summary, i)
                summary_train_loss = [tf.Summary.Value(tag='Losses', simple_value=cross_entropy_value)]
                train_writer.add_summary(tf.Summary(value=summary_train_loss), i)
                train_writer.flush()
                print("Training Loss in iteration %i: %2.6f" % (i, cross_entropy_value))

                # Optional Validation
                if self.validation_percentage > 0 and (i % self.how_often_validation == 0):
                    total_validation_loss = []
                    maximumNumberOfValidationIterations = int(
                        np.ceil(self.samplehandler.amountOfValidationSamples / self.batchsize))
                    for val_iter in range(maximumNumberOfValidationIterations):
                        var_batch_size = min(self.batchsize,
                                             self.samplehandler.amountOfValidationSamples - self.samplehandler.nextUnusedValidationSampleIndex)

                        (val_image_data,
                         val_ground_truth, _) = self.samplehandler.get_random_samples(how_many=var_batch_size,
                                                                                      purpose='valid',
                                                                                      session=sess,
                                                                                      jpeg_data_tensor=jpeg_data_tensor,
                                                                                      decoded_image_tensor=decoded_image_tensor)

                        feed_dictionary = {input_image_tensor: val_image_data}
                        val_ground_truth_ = np.asarray(val_ground_truth)

                        if (len(feed_dictionary[input_image_tensor])) == 0: break
                        for ind in range(np.shape(val_ground_truth_)[0]):
                            feed_dictionary[ground_truth_input[ind]] = val_ground_truth_[ind]

                        (validation_summary, validation_loss) = sess.run([merged, cross_entropy],
                                                                         feed_dict=feed_dictionary)
                        total_validation_loss.append(validation_loss)
                        # print("It took %2.2f seconds to process a validation batch" % ((dt.datetime.now() - t1).seconds + (dt.datetime.now() - t1).microseconds/1e6))

                    total_validation_loss = np.mean(total_validation_loss)
                    summary_val_loss = [tf.Summary.Value(tag='Losses', simple_value=total_validation_loss)]
                    val_writer.add_summary(tf.Summary(value=summary_val_loss), i)
                    val_writer.flush()
                    # print("Validation Loss: " + str(total_validation_loss))
                    print("Validation Loss in iteration %i: %2.6f" % (i, total_validation_loss))

                # Save (best) configuration
                if self.validation_percentage > 0:
                    if best_validation_loss > total_validation_loss or best_validation_loss == -1:
                        print("New best model found!")
                        train_saver.save(sess, self.log_dir + '/' + 'model.ckpt')
                        best_validation_loss = total_validation_loss
                else:
                    train_saver.save(sess, self.log_dir + '/' + 'model.ckpt')

                # parallel testing on target domain
                if self.validation_percentage > 0 and (i % self.how_often_validation == 0):
                    totalTargetLoss = []
                    maximumNumberOfValidationIterations = int(
                        np.ceil(self.samplehandlerTarget.amountOfValidationSamples / self.batchsize))
                    for val_iter in range(maximumNumberOfValidationIterations):
                        var_batch_size = min(self.batchsize,
                                             self.samplehandlerTarget.amountOfValidationSamples - self.samplehandlerTarget.nextUnusedValidationSampleIndex)

                        (val_image_data,
                         val_ground_truth, _) = self.samplehandlerTarget.get_random_samples(how_many=var_batch_size,
                                                                                            purpose='valid',
                                                                                            session=sess,
                                                                                            jpeg_data_tensor=jpeg_data_tensor,
                                                                                            decoded_image_tensor=decoded_image_tensor)

                        feed_dictionary = {input_image_tensor: val_image_data}
                        val_ground_truth_ = np.asarray(val_ground_truth)

                        if (len(feed_dictionary[input_image_tensor])) == 0: break
                        for ind in range(np.shape(val_ground_truth_)[0]):
                            feed_dictionary[ground_truth_input[ind]] = val_ground_truth_[ind]

                        (validation_summary, validation_loss) = sess.run([merged, cross_entropy],
                                                                         feed_dict=feed_dictionary)
                        totalTargetLoss.append(validation_loss)
                        # print("It took %2.2f seconds to process a validation batch" % ((dt.datetime.now() - t1).seconds + (dt.datetime.now() - t1).microseconds/1e6))

                    totalTargetLoss = np.mean(totalTargetLoss)
                    summary_target_loss = [tf.Summary.Value(tag='Losses', simple_value=totalTargetLoss)]
                    target_writer.add_summary(tf.Summary(value=summary_target_loss), i)
                    target_writer.flush()
                    # print("Validation Loss: " + str(total_validation_loss))
                    print("Target Loss in iteration %i: %2.6f" % (i, totalTargetLoss))

        # Save task_dict for subsequent functions
        np.savez(self.log_dir + r"/task_dict.npz", self.samplehandler.taskDict)

    """ROUTINES CNN"""

    def create_module_graph(self, module_spec):
        """Creates the complete computation graph, including feature extraction,
           data augmentation and classification.

        :Arguments:
          :module_spec:
              The hub.ModuleSpec for the image module being used.
        """

        """*********************BEGIN: CREATE MODULE GRAPH****************************"""
        height, width = hub.get_expected_image_size(module_spec)
        #  print('Input size of the first pre-trained layer:', height, width, 3)

        # 'outdated'' finetuning-code
        with tf.compat.v1.variable_scope("moduleLayers"):
            input_image_tensor = tf.compat.v1.placeholder(tf.float32, [None, height, width, 3], name="input_img")
            m = hub.Module(module_spec)
            bottleneck_tensor = m(input_image_tensor)

        # get variables to retrain
        help_string_1 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='moduleLayers/module/resnet_v2_152')[0].name
        help_string_2 = help_string_1.split('/')[0] + '/' + help_string_1.split('/')[1] + '/block'

        temp_choice = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=help_string_2)
        max_block = 0
        max_unit = 0
        residual_block_dict = {}
        num_residual_blocks = 0
        for variable in temp_choice:
            temp_block = int(variable.name.split('block')[1].split('/')[0])
            temp_unit = int(variable.name.split('unit_')[1].split('/')[0])
            if temp_block not in residual_block_dict.keys():
                residual_block_dict[temp_block] = [temp_unit]
            else:
                residual_block_dict[temp_block].append(temp_unit)
            if temp_block > max_block:
                max_block = temp_block
                num_residual_blocks += max_unit
                max_unit = 0
            if temp_unit > max_unit:
                max_unit = temp_unit
        num_residual_blocks += max_unit

        num_added_res_blocks = 0
        self.trainable_variables = []
        for makro_block in range(max_block, 0, -1):
            for res_block in range(max(residual_block_dict[makro_block]), 0, -1):
                if num_added_res_blocks < self.num_finetune_layers:
                    self.trainable_variables.extend(
                        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=help_string_2 + str(
                                                        makro_block) + '/unit_'
                                                          + str(res_block)))
                    num_added_res_blocks += 1
                else:
                    break

        return input_image_tensor, bottleneck_tensor

    def add_MTL_graph(self, bottleneck_tensor):
        """Adds the classification networks for multitask learning."""
        with tf.compat.v1.variable_scope("customLayers"):
            ground_truth_list = []
            batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
            assert batch_size is None, 'We want to work with arbitrary batch size.'
            with tf.name_scope('input'):
                for MTL_task in self.samplehandler.classCountDict.keys():
                    if self.multiLabelsListOfVariables is None:  # one label indice per sample
                        ground_truth_input = tf.compat.v1.placeholder(
                            tf.int64, [batch_size], name='GroundTruthInput' + MTL_task)
                    else:  # one-/multi-hot-encoding of the labels
                        if MTL_task in self.multiLabelsListOfVariables:
                            ground_truth_input = tf.compat.v1.placeholder(
                                tf.int64, [batch_size, self.samplehandler.numClassPerTask[MTL_task]],
                                name='GroundTruthInputMultiLabel' + MTL_task)
                        else:
                            ground_truth_input = tf.compat.v1.placeholder(
                                tf.int64, [batch_size, self.samplehandler.numClassPerTask[MTL_task]],
                                name='GroundTruthInputMultiClass' + MTL_task)
                    ground_truth_list.append(ground_truth_input)
                ground_truth_MTL = tuple(ground_truth_list)
                for MTL_ind, MTL_key in enumerate(self.samplehandler.classCountDict.keys()):
                    tf.compat.v1.summary.histogram("class_distribution" + MTL_key, ground_truth_MTL[MTL_ind])

            # 1.3 Initializer for the layer weights
            init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)

            # 2.1 Create a fc for all tasks together to adapt the features from the other
            #     domain.
            # 2.2 Create one classification network per classification task

            with tf.compat.v1.variable_scope("joint_layers_MTL"):
                dropout_rate_input_tensor = tf.compat.v1.placeholder_with_default(tf.constant(0.), shape=[],
                                                                                  name="dropout_rate_input")
                bottleneck_tensor = tf.nn.dropout(bottleneck_tensor, rate=dropout_rate_input_tensor,
                                                  name="dropout_layer")

                if len(self.add_fc) == 1:
                    joint_fc = tf.layers.dense(inputs=bottleneck_tensor,
                                                            units=self.add_fc[-1],
                                                            use_bias=True,
                                                            kernel_initializer=init,
                                                            activation=None,
                                                            name='output_features')
                elif len(self.add_fc) > 1:
                    for cur_fc in range(len(self.add_fc) - 1):
                        if cur_fc == 0:
                            dense_layer = tf.layers.dense(inputs=bottleneck_tensor,
                                                          units=self.add_fc[cur_fc],
                                                          use_bias=True,
                                                          kernel_initializer=init,
                                                          activation=tf.nn.relu,
                                                          name='fc_layer' + str(cur_fc) +
                                                               '_' + str(self.add_fc[cur_fc]))
                        else:
                            dense_layer = tf.layers.dense(inputs=dense_layer,
                                                          units=self.add_fc[cur_fc],
                                                          use_bias=True,
                                                          kernel_initializer=init,
                                                          activation=tf.nn.relu,
                                                          name='fc_layer' + str(cur_fc) +
                                                               '_' + str(self.add_fc[cur_fc]))
                    joint_fc = tf.layers.dense(inputs=dense_layer,
                                                            units=self.add_fc[-1],
                                                            use_bias=True,
                                                            kernel_initializer=init,
                                                            activation=None,
                                                            name='output_features')
                else:
                    joint_fc = bottleneck_tensor

            final_tensor_MTL = []
            logits_MTL = []
            # Count missing tasks for every sample, i.e. to which degree one sample is incomplete
            assert self.num_task_stop_gradient <= len(self.samplehandler.classCountDict.keys()), \
                "num_task_stop_gradient has to be smaller or equal to the number of tasks!"
            if self.num_task_stop_gradient < 0: self.num_task_stop_gradient = len(
                self.samplehandler.classCountDict.keys())
            if self.multiLabelsListOfVariables is None:  # label indices, nan=-1
                count_incomplete = tf.fill(tf.shape(ground_truth_MTL[0]), 0.0)
                for MTL_ind, MTL_task in enumerate(self.samplehandler.classCountDict.keys()):
                    temp_labels = ground_truth_MTL[MTL_ind]
                    temp_zero = tf.fill(tf.shape(temp_labels), 0.0)
                    temp_one = tf.fill(tf.shape(temp_labels), 1.0)
                    temp_incomplete = tf.where(tf.equal(temp_labels, -1), temp_one, temp_zero)
                    count_incomplete = count_incomplete + temp_incomplete
            else:  # one-/multi-hot encoded labels, nan=zeros-vector
                count_incomplete = tf.fill([tf.shape(ground_truth_MTL[0])[0]], 0.0)
                for MTL_ind, MTL_task in enumerate(self.samplehandler.classCountDict.keys()):
                    temp_labels = ground_truth_MTL[MTL_ind]
                    temp_zero = tf.fill([tf.shape(ground_truth_MTL[0])[0]], 0.0)
                    temp_one = tf.fill([tf.shape(ground_truth_MTL[0])[0]], 1.0)
                    temp_incomplete = tf.where(tf.equal(tf.math.reduce_sum(temp_labels, -1), 0), temp_one, temp_zero)
                    count_incomplete = count_incomplete + temp_incomplete
            mask_contribute = tf.math.greater_equal(tf.cast(self.num_task_stop_gradient, tf.float32), count_incomplete)

            for MTL_ind, MTL_task in enumerate(self.samplehandler.classCountDict.keys()):
                with tf.compat.v1.variable_scope("stop_incomplete_gradient_" + MTL_task):
                    # For each task, split complete from incomplete samples. That way
                    # the gradient can be stopped for incomplete samples so that they won't
                    # contribute to the update of the joint fc layer(s)
                    # temp_labels = ground_truth_MTL[MTL_ind]
                    temp_zero = tf.fill(tf.shape(joint_fc), 0.0)
                    contrib_samples = tf.where(mask_contribute, joint_fc, temp_zero)
                    nocontrib_samples = tf.stop_gradient(tf.where(mask_contribute, temp_zero, joint_fc))
                    joint_fc_ = contrib_samples + nocontrib_samples
                    tf.compat.v1.summary.histogram('activations_FC_contribute_' + str(MTL_task), contrib_samples)
                    tf.compat.v1.summary.histogram('activations_FC_nocontribute_' + str(MTL_task), nocontrib_samples)

                if MTL_task == "museum":
                    print("Gradient Reversal will be applied to museum branch!")
                    joint_fc_ = self.gradientReversal(joint_fc_)

                if self.restriction_Anne:
                    dense_0 = joint_fc_
                else:
                    dense_0 = tf.layers.dense(inputs=joint_fc_,
                                              units=100,
                                              use_bias=True,
                                              kernel_initializer=init,
                                              activation=tf.nn.relu,
                                              name='1st_fc_layer_' + str(MTL_task))

                # consider only single label classes in the logits
                num_logits = len(self.samplehandler.classPerTaskDict[MTL_task])
                logits_0 = tf.layers.dense(inputs=dense_0,
                                           units=num_logits,
                                           use_bias=True,
                                           kernel_initializer=init,
                                           activation=None,
                                           name='2nd_fc_layer_' + str(
                                               MTL_task) + '_' + str(
                                               num_logits) + '_classes')
                # all mutli-class
                if self.multiLabelsListOfVariables is None:
                    final_tensor_0 = tf.nn.softmax(logits_0,
                                                   name=self.final_tensor_name + '_' + str(MTL_task))
                # multi-class or multi-label
                else:
                    # multi-label
                    if MTL_task in self.multiLabelsListOfVariables:
                        final_tensor_0 = tf.math.sigmoid(logits_0,
                                                         name=self.final_tensor_name + '_' + str(MTL_task))
                    # multi-class
                    else:
                        final_tensor_0 = tf.nn.softmax(logits_0,
                                                       name=self.final_tensor_name + '_' + str(MTL_task))
                print("Final Tensor:", self.final_tensor_name + '_' + str(MTL_task))
                final_tensor_MTL.append(final_tensor_0)
                logits_MTL.append(logits_0)

                tf.compat.v1.summary.histogram('activations_' + str(MTL_task), final_tensor_0)

            final_tensor_MTL = tf.tuple(final_tensor_MTL, name=self.final_tensor_name)
            logits_MTL = tuple(logits_MTL)

        return ground_truth_MTL, final_tensor_MTL, logits_MTL, dropout_rate_input_tensor

    @staticmethod
    def gradientReversal(inputTensor):
        with tf.name_scope("gradient-reversal"):
            forwardPath = tf.stop_gradient(inputTensor * tf.cast(2., tf.float32))
            backwardPath = -inputTensor * tf.cast(1., tf.float32)
        return forwardPath + backwardPath

    def addOptimizer(self, loss):
        self.trainable_variables.extend(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='customLayers'))
        with tf.name_scope('L2-Loss'):
            lossL2 = tf.reduce_mean([tf.nn.l2_loss(v) for v in self.trainable_variables if 'bias' not in v.name])
        loss = loss + lossL2 * self.weight_decay

        with tf.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            grad_var_list = optimizer.compute_gradients(loss,
                                                        tf.trainable_variables())
            for (grad, var) in grad_var_list:
                tf.compat.v1.summary.histogram(var.name + '/gradient', grad)
                tf.compat.v1.summary.histogram(var.op.name, var)
            train_step = optimizer.apply_gradients(grad_var_list)

        return train_step, loss

    def image_lists_to_image_array(self, image_lists):
        """Converts an array of image lists into an array of imagess.

        :Arguments:
            :image_lists:
                It's an array containing all file names of the "image_file.txt" (the
                master file in array form). Each "image_file.txt" contains a list
                of the images to be classified.

        :Returns:
            :image_array:
                It's an array containing all images with their relative path from
                the master directory to their storage location.
        """

        var_dir = r"img_classification/"

        if not os.path.exists(self.masterfile_dir + var_dir):
            os.makedirs(self.masterfile_dir + var_dir)

        image_array = []
        for image_list in image_lists:
            im_id = open(os.path.join(self.masterfile_dir, image_list).strip("\n"), 'r')
            for line, im_line in enumerate(im_id):

                # skip header
                if line == 0: continue

                # Check if image is given as a file with path or via URL
                # Try to download when the file does not yet exist

                rel_im_path = im_line.replace('\n', '')
                if not os.path.isfile(self.masterfile_dir + rel_im_path):
                    # URL case
                    name = rel_im_path.split('/')[-1]
                    rel_im_path = var_dir + name
                    if not os.path.isfile(self.masterfile_dir + rel_im_path):
                        urllib.request.urlretrieve(im_line, self.masterfile_dir + rel_im_path)
                        try:
                            urllib.request.urlretrieve(im_line, self.masterfile_dir + rel_im_path)
                        except:
                            print(self.masterfile_dir + rel_im_path)
                            print(im_line + "can not be accessed!")
                            continue

                image_array.append(rel_im_path)
            im_id.close()
        #        1/0
        return image_array

    """READ CONFIGFILES"""

    def read_configfile(self, configfile):
        """Reads all types of configfiles and sets internal parameters"""

        control_id = open(configfile, 'r', encoding='utf-8')
        for variable in control_id:

            # Create Dataset
            if variable.split(';')[0] == 'csvfile':
                self.csvfile = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'imgsave_dir':
                self.imgsave_dir = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'minnumsamples':
                self.minnumsamples = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'retaincollections':
                self.retaincollections = variable.split(';')[1].replace(',', '') \
                                             .replace(' ', '').replace('\n', '') \
                                             .replace('\t', '').split('#')[2:]

            if variable.split(';')[0] == 'create_default_configfiles':
                self.create_default_configfiles = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'num_labeled':
                self.num_labeled = np.int(variable.split(';')[1].strip())

            # format of samples
            if variable.split(';')[0] == 'image_based_samples':
                image_based_samples = variable.split(';')[1].strip()
                if image_based_samples == 'True':
                    image_based_samples = True
                else:
                    image_based_samples = False
                self.image_based_samples = image_based_samples

            # Directories
            if variable.split(';')[0] == 'masterfile_name':
                self.masterfile_name = variable.split(';')[1].strip()
                self.masterfile_name_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'masterfile_dir':
                self.masterfile_dir = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'log_dir':
                self.log_dir = variable.split(';')[1].strip()
                self.log_dir_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'result_dir':
                self.result_dir = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'model_dir':
                self.model_dir = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'pred_gt_dir':
                self.pred_gt_dir = variable.split(';')[1].strip()

            # Network Architecture
            if variable.split(';')[0] == 'num_joint_fc_layer':
                self.num_joint_fc_layer = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'num_nodes_joint_fc':
                self.num_nodes_joint_fc = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'num_finetune_layers':
                self.num_finetune_layers = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'relevant_variables':
                self.relevant_variables = variable.split(';')[1].replace(',', '') \
                                              .replace(' ', '').replace('\n', '') \
                                              .replace('\t', '').split('#')[1:]

            # Training Specifications
            if variable.split(';')[0] == 'batchsize':
                self.batchsize = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'how_many_training_steps':
                self.how_many_training_steps = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'how_often_validation':
                self.how_often_validation = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'validation_percentage':
                self.validation_percentage = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'learning_rate':
                self.learning_rate = np.float(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'weight_decay':
                self.weight_decay = float(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'num_task_stop_gradient':
                self.num_task_stop_gradient = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'dropout_rate':
                self.dropout_rate = float(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'nameOfLossFunction':
                self.nameOfLossFunction = str(variable.split(';')[1].strip()).lower()

            # Augmentation
            if variable.split(';')[0] == 'flip_left_right':
                flip_left_right = variable.split(';')[1].strip()
                if flip_left_right == 'True':
                    flip_left_right = True
                else:
                    flip_left_right = False
                self.aug_set_dict['flip_left_right'] = flip_left_right

            if variable.split(';')[0] == 'flip_up_down':
                flip_up_down = variable.split(';')[1].strip()
                if flip_up_down == 'True':
                    flip_up_down = True
                else:
                    flip_up_down = False
                self.aug_set_dict['flip_up_down'] = flip_up_down

            if variable.split(';')[0] == 'random_shear':
                random_shear = list(map(float,
                                        variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_shear'] = random_shear

            if variable.split(';')[0] == 'random_brightness':
                random_brightness = int(variable.split(';')[1].strip())
                self.aug_set_dict['random_brightness'] = random_brightness

            if variable.split(';')[0] == 'random_crop':
                random_crop = list(map(float,
                                       variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_crop'] = random_crop

            if variable.split(';')[0] == 'random_rotation':
                random_rotation = float(variable.split(';')[1].strip()) * math.pi / 180
                self.aug_set_dict['random_rotation'] = random_rotation

            if variable.split(';')[0] == 'random_contrast':
                random_contrast = list(map(float,
                                           variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_contrast'] = random_contrast

            if variable.split(';')[0] == 'random_hue':
                random_hue = float(variable.split(';')[1].strip())
                self.aug_set_dict['random_hue'] = random_hue

            if variable.split(';')[0] == 'random_saturation':
                random_saturation = list(map(float,
                                             variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_saturation'] = random_saturation

            if variable.split(';')[0] == 'random_rotation90':
                random_rotation90 = variable.split(';')[1].strip()
                if random_rotation90 == 'True':
                    random_rotation90 = True
                else:
                    random_rotation90 = False
                self.aug_set_dict['random_rotation90'] = random_rotation90

            if variable.split(';')[0] == 'gaussian_noise':
                gaussian_noise = float(variable.split(';')[1].strip())
                self.aug_set_dict['gaussian_noise'] = gaussian_noise

        control_id.close()

    def _write_train_parameters_to_configuration_file(self):
        config = open(os.path.join(self.log_dir, "Configuration_train_model.txt"), "w")

        config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
        config.writelines(["masterfile_name; " + self.masterfile_name + "\n"])
        config.writelines(["masterfile_dir; " + self.masterfile_dir + "\n"])
        config.writelines(["log_dir; " + self.log_dir + "\n"])
        # config.writelines(["model_dir; " + self.model_dir + "\n"])
        # config.writelines(["pred_gt_dir; " + self.pred_gt_dir + "\n"])

        config.writelines(["\n****************CNN ARCHITECTURE SPECIFICATIONS**************** \n"])
        config.writelines(["num_joint_fc_layer; " + str(self.num_joint_fc_layer) + "\n"])
        config.writelines(["num_nodes_joint_fc; " + str(self.num_nodes_joint_fc) + "\n"])
        config.writelines(["num_fine_tune_layers; " + str(self.num_finetune_layers) + "\n"])

        config.writelines(["\n****************TRAINING SPECIFICATIONS**************** \n"])
        config.writelines(["batchsize; " + str(self.batchsize) + "\n"])
        config.writelines(["how_many_training_steps; " + str(self.how_many_training_steps) + "\n"])
        config.writelines(["learning_rate; " + str(self.learning_rate) + "\n"])
        config.writelines(["validation_percentage; " + str(self.validation_percentage) + "\n"])
        config.writelines(["how_often_validation; " + str(self.how_often_validation) + "\n"])
        config.writelines(["nameOfLossFunction; " + self.nameOfLossFunction + "\n"])
        config.writelines(["weight_decay; " + str(self.weight_decay) + "\n"])
        config.writelines(["num_task_stop_gradient; " + str(self.num_task_stop_gradient) + "\n"])

        config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
        config.writelines(["relevant_variables; "])
        for variable in self.relevant_variables[0:-1]:
            config.writelines(["#%s, " % str(variable)])
        config.writelines(["#%s\n" % str(self.relevant_variables[-1])])

        config.writelines(["\n****************DATA AUGMENTATION SPECIFICATIONS**************** \n"])
        # config.writelines(["random_crop; " + str(self.aug_set_dict["random_crop"]) + "\n"])
        # config.writelines(["random_rotation90; " + str(self.aug_set_dict["random_rotation90"]) + "\n"])
        # config.writelines(["gaussian_noise; " + str(self.aug_set_dict["gaussian_noise"]) + "\n"])
        # config.writelines(["flip_left_right; " + str(self.aug_set_dict["flip_left_right"]) + "\n"])
        # config.writelines(["flip_up_down; " + str(self.aug_set_dict["flip_up_down"]) + "\n"])
        config.close()
