import random


# def modify_dataset_and_raw_data_for_experiments(initial_dataset, raw_data, percentage_of_rows_to_keep, percentage_of_features_to_keep, modif_on_rows=True):
#     if modif_on_rows:
#         modified_preprocessed_data, modified_raw_data = modify_dataset_and_raw_data_with_percentage_size_to_keep(preprocessed_data, raw_data, percentage_of_rows_to_keep)
#         return modified_preprocessed_data, modified_raw_data
#     else:
#         modified_preprocessed_data = modify_dataset_with_percentage_of_features_to_keep(initial_dataset, percentage_of_features_to_keep)
#         return modified_preprocessed_data, raw_data
import numpy
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif, chi2
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def modify_dataset_and_raw_data_with_percentage_size_to_keep(initial_dataset, raw_data, dataset_size_percentage_to_keep):
    """Extract a subset of the dataset while keeping the distribution of the classes of the original dataset."""
    if dataset_size_percentage_to_keep == 100:
        return initial_dataset, raw_data
    y = raw_data['label']
    split = dataset_size_percentage_to_keep/100
    new_dataset, _, new_raw_data, _ = train_test_split(initial_dataset, raw_data,
                                                      stratify=y,
                                                      test_size=split,
                                                      random_state=RANDOM_SEED,
                                                      shuffle=True)
    assert all(raw_data.iloc[new_raw_data.index] == new_raw_data)
    return new_dataset, new_raw_data


def modify_dataset_select_features(dataset, raw_data, percentage_of_features):
    """ Select the percentage of features using ANOVA chi2
        More info: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
    """
    if percentage_of_features == 100:
        return dataset, raw_data
    y = raw_data['label']
    new_dataset = SelectPercentile(chi2, percentile=percentage_of_features).fit_transform(dataset, y)
    return new_dataset, raw_data


# def modify_datatypes(preprocessed_data, raw_data, datatype):
#     type_dictionary = {
#         'float32': numpy.single,
#         'float64': numpy.double,
#         'float128': numpy.longdouble,
#         'int16': numpy.int16,
#         'int32': numpy.intc,
#         'int64': numpy.int64,
#     }
#     modified_dataset = preprocessed_data.astype(type_dictionary[datatype])
#     return modified_dataset, raw_data
