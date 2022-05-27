
import sys
try:
    sys.path.insert(0, r"./src/")
    import silk_classification_func as sic
except:
    import silknow_image_classification as sic


# Image classifier (softmax loss)
# -------------------------------

# performs training
sic.train_model_parameter(masterfile_name="masterfile_train.txt",
                          masterfile_dir=r"./samples/",
                          log_dir=r"./classifier_softmax/",
                          num_finetune_layers=5,
                          relevant_variables=["material", "timespan", "technique", "place"],
                          learning_rate=1e-4,
                          weight_decay=1e-3,
                          nameOfLossFunction="sce",
                          multi_label_variables=None,
                          add_fc=[1024, 128]
                          )

# performs classification on test set
sic.classify_images_parameter(masterfile_name="masterfile_test.txt",
                              masterfile_dir=r"./samples/",
                              model_dir=r"./classifier_softmax/",
                              result_dir=r"./classifier_softmax/"
                              )

# evaluates the classification on test set
sic.evaluate_model_parameter(pred_gt_dir=r"./classifier_softmax/",
                             result_dir=r"./classifier_softmax/")

sic.postprocess_predictions_for_txt(input_img_centered_pred=r"./classifier_softmax/",
                                   output_dir=r"./classifier_softmax/",
                                   masterfile_dir=r"./samples/",
                                   masterfile_name="masterfile_test.txt",
                                   model_dir=r"./classifier_softmax/",
                                   path_img_data=r"../../data/img/")

# Image classifier (focal loss)
# -------------------------------

# performs training
sic.train_model_parameter(masterfile_name="masterfile_train.txt",
                          masterfile_dir=r"./samples/",
                          log_dir=r"./classifier_focal/",
                          num_finetune_layers=10,
                          relevant_variables=["material", "timespan", "technique", "place"],
                          learning_rate=1e-4,
                          weight_decay=1e-3,
                          nameOfLossFunction="focal",
                          multi_label_variables=None,
                          add_fc=[1024, 128]
                          )

# performs classification on test set
sic.classify_images_parameter(masterfile_name="masterfile_test.txt",
                              masterfile_dir=r"./samples/",
                              model_dir=r"./classifier_focal/",
                              result_dir=r"./classifier_focal/"
                              )

# evaluates the classification on test set
sic.evaluate_model_parameter(pred_gt_dir=r"./classifier_focal/",
                             result_dir=r"./classifier_focal/")

sic.postprocess_predictions_for_txt(input_img_centered_pred=r"./classifier_focal/",
                                   output_dir=r"./classifier_focal/",
                                   masterfile_dir=r"./samples/",
                                   masterfile_name="masterfile_test.txt",
                                   model_dir=r"./classifier_focal/",
                                    path_img_data=r"../../data/img/")