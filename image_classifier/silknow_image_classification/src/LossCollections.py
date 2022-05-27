import numpy as np
import tensorflow as tf


# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
# tf.nn.sigmoid_cross_entropy_with_logits(    labels=None, logits=None, name=None)

def getLossFunction(nameOfLossFunction, lossParameters):
    """Creates and returns a loss function with specific parameters.
        :Arguments:
            :nameOfLossFunction (*string*)\::
                Name of the loss function that shall be used.
            :lossParameters (*dict*)\::
                Dictionary that defines the parameters of the chosen loss.
        :Returns:
            :lossFunction:
                Function that implements the chosen loss. Each instance of lossFunction takes
                three input arguments:
                :groundtruth (*tuple*)\::
                    A tuple of groundtruth indices.
                :logits (*tuple*)\::
                    A tuple of logit outputs.
                :classCountDict (*dict*):\\
                    A dictionary containing the number of samples for every class and task.

        """
    if nameOfLossFunction == "focal":
        if "gamma" not in lossParameters.keys():
            gamma = 2.
            print("Using default value for focal loss parameter: gamma = %i" % gamma)
        else:
            gamma = lossParameters["gamma"]

        lossFunction = createInstanceOfFocalLossFunction(gamma)

    elif nameOfLossFunction == "sce":
        lossFunction = softmaxCrossEntropyLoss

    elif nameOfLossFunction == "mixed_sce":
        lossFunction = mixedSoftmaxSigmoidCrossEntropyLoss

    else:
        raise Exception("The chosen loss function %s is not available!" % nameOfLossFunction)

    return lossFunction


def createInstanceOfFocalLossFunction(gamma):
    """Creates a focal loss function with a set gamma"""
    func = lambda gt, lg, ccd: focalLoss(gt, lg, ccd, gamma)
    return func


def mixedSoftmaxSigmoidCrossEntropyLoss(groundtruth, logits, boolSigmoidActivation):
    """Computes the cross entropy for multitask learning.

    :Arguments:
        :groundtruth (*tuple*)\::
            Tuple with ground truth for each sample.
        :logits (*tuple*)\::
            Tuple with predicted classes for each sample.
        :boolSigmoidActivation (*dict*)\::
            boolSigmoidActivation[task]==True; mutli-label classification with Sigmoid activation.
            boolSigmoidActivation[task]==False; mutli-class classification with Softmax activation.
        :kwargs:
            Additional keyword arguments. Can be ignored for this function.

    :Returns:
        :cross_entropy_MTL:
            The cross entropy loss
    """
    all_cross_entropy = []
    tensor_created = False
    for MTL_ind in range(np.shape(groundtruth)[0]):
        boolSigmoid = boolSigmoidActivation[list(boolSigmoidActivation.keys())[MTL_ind]]
        temp_labels_one_hot = groundtruth[MTL_ind]
        temp_logits = logits[MTL_ind]

        if boolSigmoid:
            temp_cross_entropy = tf.compat.v1.losses.sigmoid_cross_entropy(
                multi_class_labels=temp_labels_one_hot,
                logits=temp_logits,
                reduction=tf.losses.Reduction.NONE)
            temp_cross_entropy_bs = tf.compat.v1.reduce_sum(temp_cross_entropy, -1)
        else:
            temp_cross_entropy_bs = tf.compat.v1.losses.softmax_cross_entropy(
                onehot_labels=temp_labels_one_hot,
                logits=temp_logits,
                reduction=tf.losses.Reduction.NONE)

        temp_cross_entropy = tf.reduce_sum(temp_cross_entropy_bs) / tf.cast(tf.shape(temp_labels_one_hot)[0],
                                                                            tf.float32)
        temp_cross_entropy = tf.reshape(temp_cross_entropy, [])

        # Set Cross Entropy to 0 when no labels are available for current batch and task
        temp_cross_entropy = tf.cond(tf.shape(temp_labels_one_hot)[0] < 1,
                                     lambda: 0.,
                                     lambda: temp_cross_entropy)

        all_cross_entropy.append(temp_cross_entropy)

        tf.summary.scalar("crossEntropyLoss/" + list(boolSigmoidActivation.keys())[MTL_ind],
                          temp_cross_entropy)

    cross_entropy_MTL = tf.reduce_sum(all_cross_entropy)
    tf.summary.scalar('total_cross_entropy_loss', cross_entropy_MTL)
    return cross_entropy_MTL


def softmaxCrossEntropyLoss(groundtruth, logits, classCountDict):
    """Computes the cross entropy for multitask learning.

    :Arguments:
        :groundtruth (*tuple*)\::
            Tuple with ground truth for each sample.
        :logits (*tuple*)\::
            Tuple with predicted classes for each sample.
        :classCountDict (*dict*)\::
            A dictionary containing the amount of samples per class per multi learning task.
            classCountDict[task][class] == amount of samples for this class and task.
        :kwargs:
            Additional keyword arguments. Can be ignored for this function.

    :Returns:
        :cross_entropy_MTL:
            The cross entropy loss
    """
    all_cross_entropy = []
    tensor_created = False
    for MTL_ind in range(np.shape(groundtruth)[0]):
        temp_class_count = len(classCountDict[list(classCountDict.keys())[MTL_ind]])
        temp_labels = groundtruth[MTL_ind]
        temp_logits = logits[MTL_ind]

        x = tf.fill(tf.shape(temp_labels), False)
        y = tf.fill(tf.shape(temp_labels), True)
        data_gap_ind = tf.where(tf.equal(temp_labels, -1), x, y)
        temp_labels = tf.boolean_mask(temp_labels, data_gap_ind)
        temp_logits = tf.boolean_mask(temp_logits, data_gap_ind)

        temp_labels_one_hot = tf.one_hot(indices=temp_labels,
                                         depth=temp_class_count,
                                         on_value=1.0,
                                         off_value=0.0)
        temp_cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=temp_labels_one_hot,
            logits=temp_logits,
            reduction=tf.losses.Reduction.NONE)

        temp_cross_entropy = tf.reduce_sum(temp_cross_entropy) / tf.cast(tf.shape(temp_labels), tf.float32)
        temp_cross_entropy = tf.reshape(temp_cross_entropy, [])

        # Set Cross Entropy to 0 when no labels are available for current batch and task
        temp_cross_entropy = tf.cond(tf.shape(temp_labels)[0] < 1,
                                     lambda: 0.,
                                     lambda: temp_cross_entropy)

        all_cross_entropy.append(temp_cross_entropy)

        tf.summary.scalar("crossEntropyLoss/" + list(classCountDict.keys())[MTL_ind],
                          temp_cross_entropy)

    cross_entropy_MTL = tf.reduce_sum(all_cross_entropy)
    tf.summary.scalar('total_cross_entropy_loss', cross_entropy_MTL)
    return cross_entropy_MTL


def focalLoss(groundtruth, logits, classCountDict, gamma):
    """Computes the focal loss for multitask learning.

    :Arguments:
        :groundtruth (*tuple*)\::
            Tuple with ground truth for each sample.
        :logits (*tuple*)\::
            Tuple with predicted classes for each sample.
        :classCountDict (*dict*)\::
            A dictionary containing the amount of samples per class per multi learning task.
            classCountDict[task][class] == amount of samples for this class and task.
        :gamma (*float*)\::
            Gamma parameter for the focal loss

    :Returns:
        :totalFocalLoss :
            The focal loss
    """

    totalFocalLoss = []
    numberOfTasks = np.shape(groundtruth)[0]
    for taskIndex in range(numberOfTasks):
        taskName = list(classCountDict.keys())[taskIndex]
        taskClassCountDict = classCountDict[taskName]

        taskGroundtruth = groundtruth[taskIndex]
        taskLogits = logits[taskIndex]

        # refine inputs: Only those with an available class label
        x = tf.fill(tf.shape(taskGroundtruth), False)
        y = tf.fill(tf.shape(taskGroundtruth), True)
        data_gap_ind = tf.where(tf.equal(taskGroundtruth, -1), x, y)
        temp_labels = tf.boolean_mask(taskGroundtruth, data_gap_ind)
        temp_logits = tf.boolean_mask(taskLogits, data_gap_ind)
        softmaxProbability = tf.nn.softmax(temp_logits)

        # calculate weights for the softmax activations in the loss
        classWeights = 1. / np.array(list(taskClassCountDict.values()))
        modulatingFactors = tf.pow(1. - softmaxProbability, gamma)
        focalWeights = tf.multiply(tf.cast(classWeights, tf.float32), tf.cast(modulatingFactors, tf.float32))
        temp_labels_one_hot = tf.one_hot(indices=temp_labels,
                                         depth=len(taskClassCountDict),
                                         on_value=1.0,
                                         off_value=0.0)
        focalWeights_final = tf.boolean_mask(focalWeights, temp_labels_one_hot)

        # calculate the focal loss of the current task
        task_cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=temp_labels_one_hot,
            logits=temp_logits,
            reduction=tf.compat.v1.losses.Reduction.NONE)
        taskFocalLoss = tf.map_fn(lambda input_values: tf.math.multiply(input_values[0],
                                                                        input_values[1]),
                                  (task_cross_entropy, focalWeights_final),
                                  dtype=tf.float32)
        tf.compat.v1.summary.scalar("taskFocalLoss/" + list(classCountDict.keys())[taskIndex],
                                    tf.reduce_mean(taskFocalLoss))

        # reduce to sum and scale by batchsize, because
        # if the current task has a known label for only one sample
        # this sample does not make up the complete loss,
        # but only its proportional fraction from all
        # theoretically available samples
        taskFocalLoss_avg = tf.reduce_sum(taskFocalLoss) / tf.cast(tf.shape(taskGroundtruth), tf.float32)

        taskFocalLoss_reshaped = tf.reshape(taskFocalLoss_avg, [])

        # Set Cross Entropy to 0 when no labels are available for current batch and task
        taskFocalLoss_final = tf.cond(tf.shape(temp_labels)[0] < 1,
                                      lambda: 0.,
                                      lambda: taskFocalLoss_reshaped)
        tf.summary.scalar("focalLoss/" + list(classCountDict.keys())[taskIndex], taskFocalLoss_final)

        totalFocalLoss.append(taskFocalLoss_final)

    totalFocalLoss = tf.reduce_sum(totalFocalLoss)
    tf.summary.scalar('total_focal_loss', totalFocalLoss)
    return totalFocalLoss
