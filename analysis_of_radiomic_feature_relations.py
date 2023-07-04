import numpy as np
import pandas as pd
import umap
import umap.plot
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def get_principle_components_by_PCA(features_for_PCA):
    scaler = StandardScaler()
    normalised_features = scaler.fit_transform(features_for_PCA)
    pca = PCA()
    transformed_data = pca.fit_transform(normalised_features)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_ratio_cumulative = np.cumsum(explained_variance_ratio)
    return transformed_data, explained_variance_ratio_cumulative, pca

def get_components_by_tSNE(features_for_tSNE, perplexity_value):
    normalised_features = features_for_tSNE.subtract(pd.Series.mean(features_for_tSNE, axis=0).T, axis = 'columns')
    tsne = TSNE(perplexity=perplexity_value)
    transformed_data = tsne.fit_transform(normalised_features)
    return transformed_data

def drop_voxels_with_missing_values(input_features):
    voxel_indexes = np.unique(np.where(features_patient.isnull().values)[0])
    output_features = input_features.drop(voxel_indexes, axis = 0)
    return(output_features)

def add_patient_to_the_rest(fetures_of_new_patient, patient, features_of_previous_patients):
    fetures_of_new_patient['patient_id'] = patient
    if (len(features_of_previous_patients) == 0):
        new_feature_set = fetures_of_new_patient
    else:
        new_feature_set = pd.concat([features_of_previous_patients, fetures_of_new_patient], ignore_index=True, axis=0)

    return(new_feature_set)

def plot_reduced_dimensional_space(reduced_components, patient_ids, method, acquisition ):
    color_map = {'238' : 'green', '243' : 'blue', '297' : 'yellow', '319' : 'cyan', '365' : 'magenta', '394' : 'orange', '400' : 'purple', '112' : 'red'}
    x=reduced_components[:, 0]
    y = reduced_components[:, 1]
   
    plt.figure()
    plt.scatter(x, y, c=patient_ids.map(color_map), label=np.unique(patient_ids ))
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title('tSNE')
    plt.savefig('tSNE' + acquisition + '_cerveny_hore.png')
    plt.show()
    return()

path_to_features = 'D:\\Projects\\Stroke\\Radiomics\\Heterogeneity_Features_Radiomics\\Voxel_based\\'
patients = os.listdir(path_to_features)
acquisitions = os.listdir(path_to_features + '\\' + str(patients[0]) + '\\')

for acquisition in acquisitions:
    acquisition = acquisitions[0]
    feattures_all_patients = []
    for patient in patients:
        path = path_to_features + '\\' + str(patient) + '\\' + str(acquisition) + '\\' + str(patient) + '_' + str(acquisition) + '_heterogeneity.csv'
        features_patient = pd.read_csv(path)

        # preprocessing
        features_preprocessed = drop_voxels_with_missing_values(features_patient)
        feattures_all_patients = add_patient_to_the_rest(features_preprocessed, patient, feattures_all_patients)

    # feature reduction and selection
    feattures_all_patients = feattures_all_patients[feattures_all_patients.patient_id != '243']

    features_by_PCA, explained_variance_ratio_cumulative, pca_model = get_principle_components_by_PCA(feattures_all_patients)
    plot_reduced_dimensional_space(features_by_PCA, feattures_all_patients.iloc[:,-1], 'PCA',  acquisition)
    most_variance_PCA_components = features_by_PCA[:, 0:np.where(explained_variance_ratio_cumulative > 0.9)[0][0]]
    features_by_tSNE = get_components_by_tSNE(pd.DataFrame(most_variance_PCA_components), 40)
    plot_reduced_dimensional_space(features_by_tSNE, feattures_all_patients.iloc[:,-1], 'tSNE', acquisition)
    features_by_UMAP = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(most_variance_PCA_components)
    plot_reduced_dimensional_space(features_by_UMAP, feattures_all_patients.iloc[:,-1], 'UMAP', acquisition)


