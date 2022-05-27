import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import urllib.request
import matplotlib.pyplot as plt
import cv2
import collections
import xlsxwriter


"""
Important Note:
Function calls which are marked with "#FIXED VARIABLES!" expect a fixed set of relevant variables.
If and when additional variables should be considered, those functions have to be adapted.
"""
relevant_variables = ["timespan", "place", "material", "technique"]

def createDatasetForCombinedSimilarityLoss(rawCSVFile, imageSaveDirectory, masterfileDirectory, minNumSamplesPerClass,
                         retainCollections, minNumLabelsPerSample,
                         flagDownloadImages, flagRescaleImages, fabricListFile=None, flagRuleDataset=False,
                         masterfileRules=None, flagColourAugmentDataset=False,multiLabelsListOfVariables=None):

    multiLabelsListOfVariables = translate_mutliLabelList(multiLabelsListOfVariables)

    # Create standard dataset
    # -> collection_1.txt - collection_5.txt
    createDataset(rawCSVFile, imageSaveDirectory, masterfileDirectory, minNumSamplesPerClass,
                  retainCollections, minNumLabelsPerSample,
                  flagDownloadImages, flagRescaleImages, fabricListFile, multiLabelsListOfVariables)

    # Create dataset for similarity rules (from domain experts)
    # -> contains samples mentioned in rules
    # -> dataset contains only ADDITIONAL samples that are not in standard dataset
    # -> collection_rules_1.txt - collection_rules_5.txt
    if flagRuleDataset:
        createDatasetSimilarityRules(rawCSVFile, imageSaveDirectory, masterfileDirectory,
                                     minNumSamplesPerClass, retainCollections, minNumLabelsPerSample,
                                     flagDownloadImages, flagRescaleImages, masterfileDirectory,
                                     masterfileRules, fabricListFile, "collection_rules", multiLabelsListOfVariables)

    # Create dataset for augmentation loss
    # -> dataset contains only ADDITIONAL samples that are not in standard and rules dataset
    # -> collection_colour_augment_1.txt - collection_colour_augment_5.txt
    if flagColourAugmentDataset:
        createDatasetColourAugment(masterfileDirectory, rawCSVFile, imageSaveDirectory, flagRuleDataset)


def translate_mutliLabelList(multiLabelsListOfVariables):
    if multiLabelsListOfVariables is not None:
        temp_new_list = []
        if "material" in multiLabelsListOfVariables:
            temp_new_list.append("material_group")
        if "place" in multiLabelsListOfVariables:
            temp_new_list.append("place_country_code")
        if "timespan" in multiLabelsListOfVariables:
            temp_new_list.append("time_label")
        if "technique" in multiLabelsListOfVariables:
            temp_new_list.append("technique_group")
        if "depiction" in multiLabelsListOfVariables:
            temp_new_list.append("depict_group")
        multiLabelsListOfVariables = temp_new_list
    return multiLabelsListOfVariables


def createDatasetColourAugment(masterfileDirectory, rawCSVFile, imageSaveDirectory, flagRuleDataset):
    semantic_csv = "image_data.csv"
    semantic_df = pd.read_csv(os.path.join(masterfileDirectory, semantic_csv))

    if flagRuleDataset:
        rule_csv = "image_data_rule.csv"
        additional_rule_df = pd.read_csv(os.path.join(masterfileDirectory, rule_csv))
    else:
        additional_rule_df = pd.DataFrame(columns=semantic_df.columns)

    objects_2_be_added = np.hstack((semantic_df.obj, additional_rule_df.obj))
    df_to_be_added = find_additional_obj_in_csv(rawCSVFile, objects_2_be_added)

    for col in df_to_be_added.columns:
        if col in ["place", "timespan", "material", "technique"]:
            df_to_be_added[col].values[:] = 'nan'

    if flagDownloadImages:
        downloadImages(df_to_be_added, imageSaveDirectory)
    if flagRescaleImages:
        rescaleImages(imageSaveDirectory)
    df_to_be_added = filterByExistingImages(df_to_be_added, imageSaveDirectory)

    df_to_be_added.to_csv(os.path.join(masterfileDirectory, "image_data_augment_colour.csv"))

    dataChunkList = getChunksWithSimilarClassDistributions(df_to_be_added, False)
    writeCollectionFiles(dataChunkList, masterfileDirectory, imageSaveDirectory, "collection_colour_augment")

    print("semantic (and potenitally rules)\n", len(semantic_df))
    print("rules but not semantic\n", len(additional_rule_df))
    print("additional for colour and augment only\n", len(df_to_be_added))
    print("total\n", len(semantic_df) + len(additional_rule_df) + len(df_to_be_added))


def find_additional_obj_in_csv(rawCSVFile, obj_already_there):
    all_img_df = pd.read_csv(rawCSVFile).set_index("ID")
    all_img_df = formatFieldStrings(all_img_df).fillna('nan')
    all_img_df = convertToImageBasedDataframe(all_img_df)
    all_img_df = discardImagesUsedInMultipleObjects(all_img_df)
    df_to_be_added = all_img_df[~all_img_df["obj"].isin(obj_already_there)]
    return df_to_be_added

def createDatasetSimilarityRules(rawCSVFile, imageSaveDirectory, masterfileDirectory,
                                 minNumSamplesPerClass, retainCollections, minNumLabelsPerSample,
                                 flagDownloadImages, flagRescaleImages, master_file_path_similar,
                                 masterfileRules, fabricListFile, CollectionFileBaseName, multiLabelsListOfVariables):
    dataframe = get_df_for_semantic_similarity(rawCSVFile, imageSaveDirectory, masterfileDirectory, minNumSamplesPerClass,
                                           retainCollections, minNumLabelsPerSample, flagDownloadImages,
                                           flagRescaleImages,
                                           fabricListFile, multiLabelsListOfVariables)

    similar_dict, sim_obj_list = get_uris_of_rule_lists(master_file_path_similar, masterfileRules)
    sim_obj_list = add_uri_ob_multi_img_obj(sim_obj_list, rawCSVFile)
    obj_2_be_added = find_obj_uri_not_in_dataset(sim_obj_list, dataframe)
    df_to_be_added = get_df_of_obj_only_in_rules(rawCSVFile, obj_2_be_added, multiLabelsListOfVariables)
    for col in df_to_be_added.columns:
        if col in ["place", "timespan", "material", "technique", "depiction"]:
            df_to_be_added[col].values[:] = 'nan'

    if flagDownloadImages:
        downloadImages(df_to_be_added, imageSaveDirectory)
    if flagRescaleImages:
        rescaleImages(imageSaveDirectory)
    df_to_be_added = filterByExistingImages(df_to_be_added, imageSaveDirectory)

    df_to_be_added.to_csv(os.path.join(masterfileDirectory, "image_data_rule.csv"))
    df_to_be_added = pd.read_csv(os.path.join(masterfileDirectory, "image_data_rule.csv")).set_index("ID")
    dataChunkList = getChunksWithSimilarClassDistributions(df_to_be_added, False)
    writeCollectionFiles(dataChunkList, masterfileDirectory, imageSaveDirectory, CollectionFileBaseName)


def get_df_for_semantic_similarity(rawCSVFile, imageSaveDirectory, masterfileDirectory, minNumSamplesPerClass,
                               retainCollections, minNumLabelsPerSample, flagDownloadImages, flagRescaleImages,
                               fabricListFile, multiLabelsListOfVariables):
    # preprocessRawCSVFile(rawCSVFile=rawCSVFile,
    #                      imageSaveDirectory=imageSaveDirectory,
    #                      masterfileDirectory=masterfileDirectory,
    #                      minNumSamplesPerClass=minNumSamplesPerClass,
    #                      retainCollections=retainCollections,
    #                      minNumLabelsPerSample=minNumLabelsPerSample,
    #                      flagDownloadImages=flagDownloadImages,
    #                      flagRescaleImages=flagRescaleImages,
    #                      fabricListFile = fabricListFile,
    #                      multiLabelsListOfVariables = multiLabelsListOfVariables)

    dataframeFile = os.path.join(masterfileDirectory, "image_data.csv")
    dataframe = pd.read_csv(dataframeFile).set_index("ID")
    return dataframe


def get_uris_of_rule_lists(master_file_path, master_file_similar):
    rule_dict = {}
    rule_obj_list = []
    rule_id = open(os.path.join(master_file_path, master_file_similar), 'r')
    for rule_file in rule_id:
        rule_list = []
        file_id = open(os.path.join(master_file_path, rule_file.strip()), 'r')
        for line in file_id:
            obj_uri = line.split("/")[-1].strip()
            rule_list.append(obj_uri)
            rule_obj_list.append(obj_uri)
        rule_dict[rule_file.strip()] = rule_list

    return rule_dict, np.unique(rule_obj_list)

def add_uri_ob_multi_img_obj(sim_obj_list, rawCSVFile):
    dataframe = pd.read_csv(rawCSVFile)
    dataframe = formatFieldStrings(dataframe, None).fillna('nan')  # FIXED VARIABLES!
    df_with_multi_img_obj_only = dataframe.loc[np.array(list(map(len, dataframe.img.values))) > 1]
    new_sim_obj_list = set.union(set(sim_obj_list), set(list(df_with_multi_img_obj_only["obj"])))
    return list(new_sim_obj_list)

def find_obj_uri_not_in_dataset(obj_list, dataset_df):
    obj_2_be_added = []
    for obj_uri in obj_list:
        if not any(dataset_df.obj.isin([obj_uri])):
            obj_2_be_added.append(obj_uri)
    return obj_2_be_added


def get_df_of_obj_only_in_rules(rawCSVFile, obj_2_be_added, multiLabelsListOfVariables):
    all_img_df = pd.read_csv(rawCSVFile)
    all_img_df = formatFieldStrings(all_img_df, multiLabelsListOfVariables).fillna('nan')
    all_img_df = convertToImageBasedDataframe(all_img_df)
    all_img_df = discardImagesUsedInMultipleObjects(all_img_df)
    df_to_be_added = all_img_df[all_img_df["obj"].isin(obj_2_be_added)]
    return df_to_be_added


def createDataset(rawCSVFile, imageSaveDirectory, masterfileDirectory, minNumSamplesPerClass,
                  retainCollections, minNumLabelsPerSample,
                  flagDownloadImages, flagRescaleImages, fabricListFile=None,
                  multiLabelsListOfVariables=None):
    """Creates the collection files containing samples for training and evaluation.

    :Arguments\::
            :rawCSVFile (*string*)\::
                Filename of the .csv file, which is the export of the SILKNOW knowledge graph.
            :imageSaveDirectory (*string*)\::
                Directory where the images will be downloaded.
                It has to be relative to the main software folder.
            :masterfileDirectory (*string*)\::
                Directory where the collection files and masterfile will be created.
            :minNumSamplesPerClass (*int*)\::
                Minimum number of samples for each class. Classes with fewer
                occurences will be ignored and set to unknown. If it is set to 1, all classes from the .csv file
                will be considered for the creation of the collection files.
                The same applies for mutli label combinations. Each combinaion has to occurs at least
                minNumSamplesPerClass times in case that multiple labels per variable are desired.
            :retainCollections (*list*)\::
                List of strings that defines the museums/collections that
                are to be used. Data from museums/collections
                not stated in this list will be omitted.
                If this list is empty, all museums/collections will be used.
            :minNumLabelsPerSample (*int*)\::
                Variable that indicates how many labels per sample should be available so that
                a sample is a valid sample and thus, part of the created dataset. The maximum
                number of num_labeled is 5, as five semantic variables are considered in the
                current implementation of this function. Choosing the maximum number means
                that only complete samples form the dataset. The value of num_labeled must not
                be smaller than 0.
            :flagDownloadImages (*bool*)\::
                If True, images not yet downloaded to ./imageSaveDirectory/img_unscaled/ will be downloaded.
            :flagRescaleImages (*bool*)\::
                If True, images not yet rescaled will be rescaled and save to ./imageSaveDirectory/img/.
            :fabricListFile (*string*)\::
                Name of a .txt or .csv file that lists all images depicting fabrics. If this parameter is set,
                all listed images will be included in the dataset. The file has to exist in the masterfileDirectory.
            :multiLabelsListOfVariables (*list*)\::
                List of strings describing fields of variables in the rawCSV that contain multiple labels per
                variable. All variable names listed in multiLabelsListOfVariables will contain multiple labels per variable in
                addition to the single variables. Given the labels "label_1", "label_2" and "label_3" for one variable
                of one image, the resulting collection files will contain a label in the format
                "label_1___label_2___label_3". Such merged labels will be handeled in subsequent function of the image
                processing module to perform multi-label classification/semantic similarity.
        """

    multiLabelsListOfVariables = translate_mutliLabelList(multiLabelsListOfVariables)

    preprocessRawCSVFile(rawCSVFile=rawCSVFile,
                         imageSaveDirectory=imageSaveDirectory,
                         masterfileDirectory=masterfileDirectory,
                         minNumSamplesPerClass=minNumSamplesPerClass,
                         retainCollections=retainCollections,
                         minNumLabelsPerSample=minNumLabelsPerSample,
                         flagDownloadImages=flagDownloadImages,
                         flagRescaleImages=flagRescaleImages,
                         fabricListFile=fabricListFile,
                         multiLabelsListOfVariables=multiLabelsListOfVariables)

    dataframeFile = os.path.join(masterfileDirectory, "image_data.csv")
    ensureClassDistributions = True if minNumSamplesPerClass > 1 else False

    aggregateClassesAndWriteCollectionFiles(dataframeFile=dataframeFile,
                                            imageSaveDirectory=imageSaveDirectory,
                                            masterfileDirectory=masterfileDirectory,
                                            ensureClassDistributions=ensureClassDistributions,
                                            CollectionFileBaseName = "collection")


def preprocessRawCSVFile(rawCSVFile, imageSaveDirectory, masterfileDirectory, minNumSamplesPerClass,
                         retainCollections, minNumLabelsPerSample,
                         flagDownloadImages, flagRescaleImages, fabricListFile, multiLabelsListOfVariables):
    """ Preprocesses the raw csv file (export from knowledge graph).

    :Arguments\::
            :rawCSVFile (*string*)\::
                Filename of the .csv file which is the export of the SILKNOW knowledge graph.
            :imageSaveDirectory (*string*)\::
                Directory where the images will be downloaded.
                It has to be relative to the main software folder.
            :masterfileDirectory (*string*)\::
                Directory where the collection files and masterfile will be created.
            :minNumSamplesPerClass (*int*)\::
                Minimum number of samples for each class. Classes with fewer
                occurences will be ignored and set to unknown. If it is set to 1, all classes from the .csv file
                will be considered for the creation of the collection files.
            :retainCollections (*list*)\::
                List of strings that defines the museums/collections that
                are to be used. Data from museums/collections
                not stated in this list will be omitted.
                If this list is empty, all museums/collections will be used.
            :minNumLabelsPerSample (*int*)\::
                Variable that indicates how many labels per sample should be available so that
                a sample is a valid sample and thus, part of the created dataset. The maximum
                number of num_labeled is 5, as five semantic variables are considered in the
                current implementation of this function. Choosing the maximum number means
                that only complete samples form the dataset. The value of num_labeled must not
                be smaller than 0.
            :flagDownloadImages (*bool*)\::
                If True, images not yet downloaded to ./imageSaveDirectory/img_unscaled/ will be downloaded.
            :flagRescaleImages (*bool*)\::
                If True, images not yet rescaled will be rescaled and save to ./imageSaveDirectory/img/.
            :fabricListFile (*string*)\::
                Name of a .txt or .csv file that lists all images depicting fabrics. If this parameter is set,
                all listed images will be included in the dataset. The file has to exist in the masterfileDirectory.
        """

    dataframe = pd.read_csv(rawCSVFile, delimiter="\t")
    museum_overview = np.unique(dataframe.museum)
    print("museums: ", museum_overview)
    dataframe = formatFieldStrings(dataframe, multiLabelsListOfVariables).fillna('nan')  # FIXED VARIABLES!
    dataframe = convertToImageBasedDataframe(dataframe)  # FIXED VARIABLES!
    dataframe = discardImagesUsedInMultipleObjects(dataframe)
    dataframe = filterByMinNumSamplesPerClass(dataframe, minNumSamplesPerClass)  # FIXED VARIABLES!
    dataframe = filterByFabricsAndNonfabrics(dataframe, masterfileDirectory, retainCollections, fabricListFile)
    dataframe = filterByMinNumLabelsPerSample(dataframe, minNumLabelsPerSample)  # FIXED VARIABLES!

    if flagDownloadImages:
        downloadImages(dataframe, imageSaveDirectory)
    if flagRescaleImages:
        rescaleImages(imageSaveDirectory)
    dataframe = filterByExistingImages(dataframe, imageSaveDirectory)

    # The two following functions need to be called again because the amount of data
    # might have changed due to malfunctioning download links or corrupted image data
    dataframe = filterByMinNumSamplesPerClass(dataframe, minNumSamplesPerClass)  # FIXED VARIABLES!
    dataframe = filterByMinNumLabelsPerSample(dataframe, minNumLabelsPerSample)  # FIXED VARIABLES!
    dataframe = replaceSpacesInLabelnames(dataframe)  # FIXED VARIABLES!

    writeClassAggregationFiles(dataframe, masterfileDirectory)  # FIXED VARIABLES!
    writeMuseumStatisticsFile(dataframe, masterfileDirectory)  # FIXED VARIABLES

    dataframe.to_csv(os.path.join(masterfileDirectory, "image_data.csv"))


def aggregateClassesAndWriteCollectionFiles(dataframeFile, imageSaveDirectory,
                                            masterfileDirectory, ensureClassDistributions,
                                            CollectionFileBaseName):
    """ Applies class aggregation and writes collection files.

    :Arguments\::
            :dataframeFile (*string*)\::
                Filename of the .csv file which is the output of the function preprocessRawCSVFile.
            :imageSaveDirectory (*string*)\::
                Directory where the images will be downloaded.
                It has to be relative to the main software folder.
            :masterfileDirectory (*string*)\::
                Directory where the collection files and masterfile will be created.
            :ensureClassDistributions (*bool*)\::
                If True, the class distribution will be approximately the same across
                the collection files.
        """
    dataframe = pd.read_csv(dataframeFile).set_index("ID")
    dataframe = aggregateClasses(dataframe, aggregationFilesDirectory=masterfileDirectory)  # FIXED VARIABLES!
    dataChunkList = getChunksWithSimilarClassDistributions(dataframe, ensureClassDistributions)  # FIXED VARIABLES!

    printLabelStatistics(dataframe)

    writeCollectionFiles(dataChunkList, masterfileDirectory, imageSaveDirectory, CollectionFileBaseName)  # FIXED VARIABLES!
    writeMasterfiles(masterfileDirectory)


# FIXED VARIABLES!
def formatFieldStrings(dataframe, multiLabelsListOfVariables):
    # obj auf URI kürzen
    dataframe.obj = dataframe.obj.apply(lambda x: x.split("/")[-1])

    # Museum auf Name kürzen
    dataframe.museum = dataframe.museum.apply(
        lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(",")[0].split("/")[-1])

    # URLs und Bildnamen zu Listen konvertieren
    dataframe.deeplink = dataframe.deeplink.apply(
        lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    dataframe.img = dataframe.img.apply(lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    dataframe.img = dataframe.img.apply(lambda x: list(map(lambda y: y.split("/")[-1], x)))

    # Arrange entries into lists
    dataframe.place_country_code = dataframe.place_country_code.apply(
        lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    bool_place_multi = getBoolMultiLabel(multiLabelsListOfVariables, "place_country_code")
    dataframe.place_country_code = dataframe.place_country_code.apply(lambda x: x[0] if len(x) == 1 else
            (createMultiLabelString(sorted(x)) + sorted(x)[-1] if bool_place_multi else 'nan')).apply(
        lambda x: np.nan if x == 'nan' else x)

    dataframe.time_label = dataframe.time_label.apply(
        lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    bool_time_multi = getBoolMultiLabel(multiLabelsListOfVariables, "time_label")
    dataframe.time_label = dataframe.time_label.apply(lambda x: x[0] if len(x) == 1 else
            (createMultiLabelString(sorted(x)) + sorted(x)[-1] if bool_time_multi else 'nan')).apply(
        lambda x: np.nan if x == 'nan' else x)

    dataframe.technique_group = dataframe.technique_group.apply(
        lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    bool_technique_multi = getBoolMultiLabel(multiLabelsListOfVariables, "technique_group")
    dataframe.technique_group = dataframe.technique_group.apply(lambda x: x[0] if len(x) == 1 else
        (createMultiLabelString(sorted(x)) + sorted(x)[-1] if bool_technique_multi else 'nan'))
    dataframe.technique_group = dataframe.technique_group.apply(
        lambda x: str(x).split("/")[-1] if not x == 'nan' else np.nan)

    dataframe.material_group = dataframe.material_group.apply(
        lambda x: str(x).strip("[").strip("]").strip().replace("', '", ",").strip("''").split(","))
    bool_material_multi = getBoolMultiLabel(multiLabelsListOfVariables, "material_group")
    dataframe.material_group = dataframe.material_group.apply(lambda x: x[0] if len(x) == 1 else
        (createMultiLabelString(sorted(x)) + sorted(x)[-1] if bool_material_multi else 'nan'))
    dataframe.material_group = dataframe.material_group.apply(lambda x: x.split("/")[-1] if not x == 'nan' else np.nan)

    # dataframe.depict_group = dataframe.depict_group.apply(
    #     lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    # bool_depict_multi = getBoolMultiLabel(multiLabelsListOfVariables, "depict_group")
    # dataframe.depict_group = dataframe.depict_group.apply(lambda x: x[0] if len(x) == 1 else
    #     (createMultiLabelString(sorted(x)) + sorted(x)[-1] if bool_depict_multi else 'nan'))
    # dataframe.depict_group = dataframe.depict_group.apply(lambda x: x.split("/")[-1] if not x == 'nan' else np.nan)

    # # split type urls
    # dataframe.category_group = dataframe.category_group.apply(
    #     lambda x: str(x).strip("[").strip("]").replace("', '", ",").strip("''").split(',')
    # )
    # # get type identifyer
    # dataframe.category_group = dataframe.category_group.apply(
    #     lambda x: x[0].split("/")[-1] if len(x) == 1 else [y.split("/")[-1] for y in x])
    # retain "fabrics" only
    # dataframe.category_group = dataframe.category_group.apply(
    #     lambda x: (x if x == "fabrics" else "nan") if isinstance(x, str) else [(y if y == "fabrics" else "nan") for y in x])

    return dataframe


def getBoolMultiLabel(multiLabelsListOfVariables, variableName):
    boolMultiLabel = False
    if not multiLabelsListOfVariables is None:
        if variableName in multiLabelsListOfVariables:
            boolMultiLabel = True
    return boolMultiLabel


def createMultiLabelString(x):
    multiLabelString = ""
    for labelString in x[0:-1]:
        multiLabelString = multiLabelString + labelString + "___"
    # multiLabelString = multiLabelString + x[-1]

    return multiLabelString


# FIXED VARIABLES!
def convertToImageBasedDataframe(dataframe):
    # Vorverarbeitung für Records, ein Record pro Bild
    totallist = []
    counter = 0
    counter_img = 0
    error_file = open("unsolved_image_types.txt", "w")
    for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        varlist = [[row.museum + "__" + row.obj + "__" + URL.split('/')[-1],
                    row.museum,
                    row.obj,
                    URL,
                    row.place_country_code,
                    row.time_label,
                    row.material_group,
                    row.technique_group] for URL in row.deeplink if not URL == 'nan']
        totallist += varlist
        # if isinstance(row.category_group, str):
        #     if row.category_group == "fabrics":
        #         totallist += varlist
        #         counter_img += len(row.deeplink)

    error_file.close()
    tl = np.asarray(totallist).transpose()

    # Erstelle Datensatz
    data = pd.DataFrame({'ID': tl[0],
                         'museum': tl[1],
                         'obj': tl[2],
                         'URL': tl[3],
                         'place': tl[4],
                         'timespan': tl[5],
                         'material': tl[6],
                         'technique': tl[7]}).replace([""], np.nan).replace("nan", np.nan)

    # shuffle data
    data = data.sample(frac=1)

    return data


def discardImagesUsedInMultipleObjects(dataframe):
    # discard all images that are used in multiple objects
    counts = dataframe.URL.value_counts(dropna=False).to_list()
    names = dataframe.URL.value_counts(dropna=False).index.to_list()
    unduplicate_images = []
    for c, n in zip(counts, names):
        if not c > 1:
            unduplicate_images.append(n)
    dataframe = dataframe[dataframe.URL.isin(unduplicate_images)]

    return dataframe


# FIXED VARIABLES!
def filterByMinNumSamplesPerClass(dataframe, minNumSamplesPerClass):
    # Throw out (i.e. set to nan) all values occuring fewer than 150 times
    for c in relevant_variables:
        names = dataframe[c].value_counts().index.tolist()
        count = dataframe[c].value_counts().tolist()
        for na, co in zip(names, count):
            if co < minNumSamplesPerClass:
                dataframe[c] = dataframe[c].replace(na, np.nan)
    return dataframe


def filterByFabricsAndNonfabrics(dataframe, masterfileDirectory, retainCollections, fabricListFile):
    if fabricListFile is None:
        fabricList = []
    else:
        fabricList = pd.read_csv(os.path.join(masterfileDirectory, fabricListFile), header=None)[0].tolist()
    dataframe = dataframe.set_index("ID")
    dataframeMuseums = dataframe[dataframe.museum.isin(retainCollections)]
    dataframeFabrics = dataframe.loc[fabricList]
    dataframeFabrics = dataframeFabrics[dataframeFabrics["museum"].notna()]

    dataframeMerged = pd.concat([dataframeMuseums,dataframeFabrics])
    dataframeMerged = dataframeMerged[~dataframeMerged.index.duplicated(keep='first')]

    return dataframeMerged.reset_index()


# FIXED VARIABLES!
def filterByMinNumLabelsPerSample(dataframe, minNumLabelsPerSample):
    # count NaNs
    dataframe['nancount'] = dataframe[relevant_variables].isnull().sum(axis=1)

    dataframe = dataframe[dataframe.nancount < len(relevant_variables) - minNumLabelsPerSample + 1]

    return dataframe


def downloadImages(dataframe, imageSaveDirectory):
    # savepath = "./img_unscaled/"
    # checkpath = "./img_unscaled/"
    savepath = checkpath = imageSaveDirectory + r"img_unscaled/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    deadlinks = 0
    deadlist = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe.index)):

        # Skip record if image already exists
        if os.path.isfile(checkpath + row.ID): continue

        # Try to download from URL until one URL works
        url = row.URL
        try:
            urllib.request.urlretrieve(url, savepath + row.ID)
        except:
            deadlinks += 1
            deadlist += [url]
    if deadlinks > 0:
        print("In total,", deadlinks, "records have no functioning image link!")
        for deadurl in deadlist:
            print(deadurl)


def rescaleImages(imageSaveDirectory):
    # Rescaling of downloaded images
    imgpath_load = imageSaveDirectory + r"/img_unscaled/"
    imgpath_save = imageSaveDirectory + r"img/"
    if not os.path.exists(imgpath_save):
        os.makedirs(imgpath_save)

    imglist = os.listdir(imgpath_load)
    print("\nTotal number of images: ", len(imglist))

    # for img_file in tqdm(imglist):
    deadlist_load = []
    deadlist_scale = []
    print("\nIterating over images (download): ")
    for img_file in tqdm(imglist, total=len(imglist)):

        # Skip images that are already scaled
        if os.path.exists(imgpath_save + img_file): continue

        # Try to open images, skip else
        try:
            img = plt.imread(imgpath_load + img_file)
        except:
            deadlist_load += [img_file]
            continue

        try:
            # get dimensions of image
            if len(img.shape) == 2:
                width, heigth = img.shape
                img_new = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
                img_new[:, :, 0], img_new[:, :, 1], img_new[:, :, 2] = img, img, img
                img = img_new
            elif len(img.shape) == 3:
                width, heigth, _ = img.shape

            smaller_side = np.minimum(heigth, width)
            scale_factor = 448. / smaller_side

            # If Downscaling, apply gaussian blur
            if scale_factor < 1.:
                sigma = 1. / scale_factor
                kernelsize = int(sigma * 6) + (1 - (int(sigma * 6) % 2))
                img = cv2.GaussianBlur(img, (kernelsize, kernelsize), sigma)

            img_new = cv2.resize(img, (int(heigth * scale_factor), int(width * scale_factor)),
                                 interpolation=cv2.INTER_CUBIC)
            plt.imsave(imgpath_save + img_file, img_new)
        except:
            # print("Fehler beim Skalieren/Speichern:"+img_file)
            deadlist_scale += [img_file]

    if len(deadlist_load) > 0:
        print("\nThe following images could not be opened: ")
        for deadload in deadlist_load:
            print(deadload)
    if len(deadlist_scale) > 0:
        print("\nThe following images could not be scaled: ")
        for _ in deadlist_scale:
            print(deadlist_scale)


def filterByExistingImages(dataframe, imageSaveDirectory):
    mypath = imageSaveDirectory + r"img/"
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    onlyfiles = list(dict.fromkeys(onlyfiles))
    intersection_set = list(set.intersection(set(onlyfiles), set(list(dataframe["ID"]))))
    missing_list = [missing for missing in list(dataframe["ID"]) if missing not in onlyfiles]
    missing_df = pd.DataFrame({"images": missing_list})
    missing_df.to_csv("missing_images.csv")
    dataframe = dataframe.set_index("ID").loc[[f for f in intersection_set]]

    return dataframe


# FIXED VARIABLES!
def replaceSpacesInLabelnames(dataframe):
    for rv in relevant_variables:
        class_labels = dataframe[rv].unique()
        for cl in class_labels:
            if " " in str(cl):
                changed_label = cl.replace(" ", "_")
                dataframe = dataframe.replace(cl, changed_label)

    return dataframe


# FIXED VARIABLES!
def writeClassAggregationFiles(dataframe, masterfileDirectory):
    for variable in relevant_variables:
        filename = "class_aggregation_" + str(variable) + ".txt"
        writeFile = open(os.path.join(masterfileDirectory, filename), "w")

        class_labels = dataframe[variable].fillna("nan").unique()
        writeFile.writelines([str(c) + " ; " + str(c) + "\n" for c in class_labels if not c == "nan"])


# FIXED VARIABLES!
def aggregateClasses(dataframe, aggregationFilesDirectory):
    for variable in relevant_variables:
        filename = "class_aggregation_" + str(variable) + ".txt"
        readFile = open(os.path.join(aggregationFilesDirectory, filename), "r", encoding="utf-8")

        for line in readFile:
            originalLabel = line.split(";")[0].strip()
            newLabel = line.split(";")[1].strip()
            dataframe[variable] = dataframe[variable].replace(originalLabel, newLabel)
        readFile.close()
    return dataframe


# FIXED VARIABLES!
def getChunksWithSimilarClassDistributions(dataframe, ensureClassDistributions):
    # get all unique class labels
    unique_labels_dict = {}
    for var in relevant_variables:
        unique_labels_dict[var] = dataframe[var].unique()

    # check that all class labels appear in every chunk
    random_chunk_split_is_ok = True
    try_counter = 0
    dataChunkList = []
    while True:
        assert try_counter < 10, "No valid data split was found in 10 tries, aborting..."

        dataframe = dataframe.sample(frac=1)
        chunked_objects = getChunksWithSimilarMuseumDistributions(dataframe)

        for chunk in chunked_objects:
            df_chunk = dataframe[dataframe.obj.isin(chunk)]
            for var in relevant_variables:
                unique_labels_in_chunk_var = df_chunk[var].unique()
                class_labels_are_matching = collections.Counter(unique_labels_in_chunk_var) == collections.Counter(
                    unique_labels_dict[var])
                random_chunk_split_is_ok = random_chunk_split_is_ok and class_labels_are_matching
            dataChunkList.append(df_chunk)
            # random_chunk_split_is_ok=True##############################
        if not random_chunk_split_is_ok and ensureClassDistributions:
            try_counter += 1
            print("Unlucky data split, retrying with new random split...")
            dataChunkList = []
        else:
            break

    return dataChunkList


def getChunksWithSimilarMuseumDistributions(dataframe):
    museums = dataframe.museum.unique()
    chunked_objects_museums = [np.array([]) for _ in range(5)]
    for m in museums:
        chunked_objects_museum = [dataframe[dataframe.museum == m].obj.unique()[i::5] for i in range(5)]
        chunked_objects_museums = [np.concatenate((chunked_objects_museums[i], chunked_objects_museum[i]), axis=0)
                                   for i in range(5)]
    return chunked_objects_museums


# FIXED VARIABLES!
def writeCollectionFiles(dataChunkList, masterfileDirectory, imageSaveDirectory, CollectionFileBaseName):
    variable_list = relevant_variables + ["museum"]
    for i, chunk in enumerate(dataChunkList):
        collection = open(masterfileDirectory + CollectionFileBaseName + "_" + str(i + 1) + ".txt", "w")
        #        string = ["#"+name+"\t" for name in list(image_data)[1:]]
        string = ["#" + name + "\t" for name in variable_list]
        collection.writelines(['#image_file\t'] + string + ["\n"])

        for index, row in chunk.iterrows():
            imagefile = str(row.name) + ".jpg\t" if ".jpg" not in str(row.name) else str(row.name) + "\t"

            string = [(str(row[label]) + "\t") for label in variable_list]

            collection.writelines([os.path.relpath(imageSaveDirectory, masterfileDirectory) + r"/img/" + imagefile] + string + ["\n"])

        collection.close()


def writeMasterfiles(masterfileDirectory):
    # Write collection files to masterfile and save it in the same path
    master = open(masterfileDirectory + "Masterfile.txt", "w+")
    for i in range(5):
        master.writelines(["collection_"] + [str(i + 1)] + [".txt\n"])
    master.close()

    masterTrain = open(masterfileDirectory + "Masterfile_train.txt", "w+")
    for i in range(4):
        masterTrain.writelines(["collection_"] + [str(i + 1)] + [".txt\n"])
    masterTrain.close()

    masterTest = open(masterfileDirectory + "Masterfile_test.txt", "w+")
    masterTest.writelines(["collection_5.txt\n"])
    masterTest.close()


# FIXED VARIABLES!
def printLabelStatistics(dataframe):
    # Print label statistics
    classStructures = {}
    for v in relevant_variables:
        print("Classes for variable", v)
        print(dataframe[v].value_counts(dropna=False))
        labels = dataframe[v].unique()
        classStructures[v] = labels[~pd.isnull(labels)]
        print("\n")

    # print label statistics per collection
    for c in dataframe.museum.unique():
        vardf = dataframe[dataframe.museum == c]
        for v in relevant_variables:
            print("Classes for variable", v, " in museum", c, ":")
            print(vardf[v].value_counts(dropna=False))
            print("\n")

        print("\n")


# FIXED VARIABLES!
def writeMuseumStatisticsFile(df, masterfileDirectory):
    df = df.fillna("nan")
    workbook = xlsxwriter.Workbook(os.path.join(masterfileDirectory,'museumStatistics.xlsx'))

    propertyList = ["place", "timespan", "technique", "material"]
    museumList = list(df["museum"].unique())
    museumList.sort()
    bold = workbook.add_format({'bold': True})

    for prop in propertyList:
        worksheet = workbook.add_worksheet(prop)
        labelList = list(df[prop].unique())
        labelList.sort()

        # make sure that nan is always the first column
        if "nan" in labelList:
            labelList.remove("nan")
            labelList = ["nan"] + labelList

        # write label names to worksheet
        for column, label in enumerate(labelList):
            worksheet.write(0, column + 1, str(label), bold)

        for row, museum in enumerate(museumList):
            worksheet.write(row + 1, 0, museum, bold)
            for column, label in enumerate(labelList):
                labelCount = sum(df[df["museum"] == museum][prop] == label)
                labelCount = labelCount if labelCount > 0 else None
                worksheet.write(row + 1, column + 1, labelCount)

        worksheet.write(len(museumList) + 2, 0, "TOTAL", bold)
        for column, label in enumerate(labelList):
            labelCount = sum(df[prop] == label)
            labelCount = labelCount if labelCount > 0 else None
            worksheet.write(len(museumList) + 2, column + 1, labelCount)

    workbook.close()

