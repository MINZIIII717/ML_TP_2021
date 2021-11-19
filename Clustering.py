import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px


# Sum of distance for elbow method
kmeans_elbowDistance = {}

# For Silhouette
Kmeans_Sil = {}
GMM_Sil = {}
Meanshift_Sil = {}
DBScan_Sil = {}

# For Purity
Kmeans_pur = {}
GMM_pur = {}
Meanshift_pur = {}
DBScan_pur = {}


def main():
    # Hyper Parameter tuning.
    K_means_parameter = list(range(3, 15, 1))
    DBScan_parameter = {'eps': [0.1, 0.2, 0.5, 5, 10, 20, 50, 100], 'min_sample': [10, 20, 50, 100, 200, 500]}
    Meanshift_parameter = [None, 1.0, 2.0, 10, 20, 50, 100]
    GMM_parameter = [0, 1.0, 2.0, 10, 20, 30, 50]

    print("1. Data Load & Preprocessing")
    data = pd.read_csv('adult.csv')  # load dataset
    # 1. Change the ? result to NaN
    data = data.replace('?', np.NaN)

    # 2. Drop the NaN values (row)
    data = data.dropna(axis=0)

    # 3. Drop columns (education, capital-gain, capital-loss)
    data.drop(['education', 'capital-gain', 'capital-loss'], axis=1, inplace=True)

    # 4. Change "native-contry" values to binary result
    # "United-States" : 1, not "United-States" : 0
    data["native-country"] = data["native-country"].apply(nativeCountry)

    # 5. Change "income" values to binary result
    # <=50k : 2, >50k :1
    data["income"] = data["income"].apply(income)
    data["income"] = data["income"].apply(pd.to_numeric)

    # 6. Change educational number to three sector
    # <10 : 1, 10~13 : 2, >13 :3
    data["educational-num"] = data["educational-num"].mask(data["educational-num"] < 10, 1)
    data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 10, 2)
    data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 11, 2)
    data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 12, 2)
    data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 13, 2)
    data["educational-num"] = data["educational-num"].mask(data["educational-num"] > 13, 3)
    pd.set_option('display.max_columns', None)

    print("2. Labeling Income")
    Income = pd.DataFrame(data["income"])
    data = data.drop(columns=["income"])
    Income['label'] = pd.cut(Income["income"], 10)

    print("3. User choose using Columns")   ## In process 3 and 4, users can choose what attribute will used in Clustering.
    data = data.drop(columns=[])

    print("4. Divide Scaling list & Encoding List")    ## Except Droped data, Put Encoding attributes and Scaling attributes in turn.
    pre_processed = Preprocessing(data,
                                ["occupation", "workclass","native-country","marital-status","relationship","race","gender"],
                                ["age", "fnlwgt", "educational-num","hours-per-week"])

    print("5. Make clustering")
    for preprocess, result in pre_processed.items():
        Best_Combination(preprocess, result, K_means_parameter, DBScan_parameter, Meanshift_parameter,
                         Income['label'])

    print("6. Result")
    ## Check sum of distance for elbow method
    ShowPlot("KMeans_distance", kmeans_elbowDistance, K_means_parameter)

    ## Silhouette
    ShowPlot("KMeans_silhouette", Kmeans_Sil, K_means_parameter)
    ShowPlot("EM_silhouette", GMM_Sil, K_means_parameter)
    ShowPlot("DBSCAN_silhouette", DBScan_Sil, DBScan_parameter['eps'])
    ShowPlot("MeanShift_distance", Meanshift_Sil, GMM_parameter)

    preprocess, result = FindBestResult(Kmeans_Sil)
    print("K-means best silhouette : ", result, preprocess)
    preprocess, result = FindBestResult(GMM_Sil)
    print("EM best silhouette : ", result, preprocess)
    preprocess, result = FindBestResult(DBScan_Sil)
    print("DBSCAN best silhouette : ", result, preprocess)
    preprocess, result = FindBestResult(Meanshift_Sil)
    print("MeanShift best silhouette : ", result, preprocess)

    # purity
    ShowPlot("KMeans_purity", Kmeans_pur, K_means_parameter)
    ShowPlot("EM_purity", GMM_pur, K_means_parameter)
    ShowPlot("DBSCAN_purity", DBScan_pur, DBScan_parameter['eps'])
    ShowPlot("MeanShift_purity", Meanshift_pur, GMM_parameter)

    preprocess, result = FindBestResult(Kmeans_pur)
    print("K-means best purity : ", result, preprocess)
    preprocess, result = FindBestResult(GMM_pur)
    print("K-means best purity : ", result, preprocess)
    preprocess, result = FindBestResult(DBScan_pur)
    print("DBSCAN best purity : ", result, preprocess)
    preprocess, result = FindBestResult(Meanshift_pur)
    print("MeanShift best purity : ", result, preprocess)


def income(x):
    if x == '<=50K':
        return x.replace(x, "0")
    return "1"


def nativeCountry(x):
    if x != "United-States":
        return str(x).replace(str(x), "0")
    return "1"


# for one-hot-encoding
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


def Preprocessing(data, encode_list, scale_list):

    # scaler
    scaler_stndard = preprocessing.StandardScaler()
    scaler_MM = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scaler_maxabs = preprocessing.MaxAbsScaler()
    scaler_normalize = preprocessing.Normalizer()
    scalers = [None, scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]
    scalers_name = ["original", "standard", "minmax", "robust", "maxabs", "normalize"]

    # encoder
    encoder_ordinal = preprocessing.OrdinalEncoder()
    # one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
    encoders_name = ["ordinal", "one-hot"]

    # result box
    result_dictionary = {}
    i = 0

    if encode_list == []:
        for scaler in scalers:
            if i == 0:  # not scaling
                result_dictionary[scalers_name[i]] = data.copy()

            else:
                # ===== scalers
                result_dictionary[scalers_name[i]] = data.copy()
                result_dictionary[scalers_name[i]][scale_list] = scaler.fit_transform(data[scale_list])  # scaling
            i = i + 1
        return result_dictionary

    for scaler in scalers:
        if i == 0:  # not scaling
            result_dictionary[scalers_name[i] + "_ordinal"] = data.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
                data[encode_list])
            result_dictionary[scalers_name[i] + "_onehot"] = data.copy()
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
                                                                        encode_list)

        else:
            # ===== scalers + ordinal encoding
            result_dictionary[scalers_name[i] + "_ordinal"] = data.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][scale_list] = scaler.fit_transform(
                data[scale_list])  # scaling
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
                data[encode_list])  # encoding

            # ===== scalers + OneHot encoding
            result_dictionary[scalers_name[i] + "_onehot"] = data.copy()
            result_dictionary[scalers_name[i] + "_onehot"][scale_list] = scaler.fit_transform(
                data[scale_list])  # scaling
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
                                                                        encode_list)  # encoding

        i = i + 1

    return result_dictionary


def Best_Combination(preprocessing_name, Data, K_means_Parameter, DBSCAN_list, MeanShift_list, EM_Parameter):
    # n_cluster : list number of cluster (use in Kmeans, GMM)
    # DBSCAN_list : list of DBSCAN parameters (eps, min_sample)
    # MeanShift_list : list of MeanShift parameters (bandwidth)

    pca = PCA(n_components=2)
    clms = Data.columns

    # KMeans
    print("Kmeans")
    Kmean_Distance = []
    Kmeans_Sil_result = []
    Kmeans_pur_result = []

    for k in K_means_Parameter:
        df_feature_pca = Data[clms]
        df_feature_pca = pca.fit_transform(df_feature_pca)
        df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])

        # arr = df_feature_pca[["PC1", "PC2"]]
        kmeans = KMeans(n_clusters=k).fit(df_feature_pca)
        # sum of distance for elbow methods
        Kmean_Distance.append(kmeans.inertia_)
        # silhouette (range -1~1)
        Kmeans_Sil_result.append(silhouette_score(df_feature_pca, kmeans.labels_, metric='euclidean'))
        # purity
        Kmeans_pur_result.append(purity_score(EM_Parameter, kmeans.labels_))
        label = kmeans.labels_
        # Visualization
        fig = px.scatter(
            df_feature_pca,
            x=df_feature_pca["PC1"],
            y=df_feature_pca["PC2"],
            color=label,
            title="KMeans"
        )
        #fig.show()

    kmeans_elbowDistance[preprocessing_name] = Kmean_Distance
    Kmeans_Sil[preprocessing_name] = Kmeans_Sil_result
    Kmeans_pur[preprocessing_name] = Kmeans_pur_result

    print("EM")
    GMM_Sil_result = []
    GMM_Pur_result = []

    for k in K_means_Parameter:
        # Use PCA, visualization
        df_feature_pca = Data[clms]
        df_feature_pca = pca.fit_transform(df_feature_pca)
        df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])

        gmm = GaussianMixture(n_components=k)
        labels = gmm.fit_predict(df_feature_pca)

        # silhouette
        GMM_Sil_result.append(silhouette_score(df_feature_pca, labels, metric='euclidean'))

        # purity
        GMM_Pur_result.append(purity_score(EM_Parameter, labels))

        fig = px.scatter(
                    df_feature_pca,
                    x=df_feature_pca["PC1"],
                    y=df_feature_pca["PC2"],
                    color=labels,
                    title="EM"
                )
        #fig.show()

    GMM_Sil[preprocessing_name] = GMM_Sil_result
    GMM_pur[preprocessing_name] = GMM_Pur_result


    # DBSCAN
    print("DBScan")
    DBScan_Sil_result = []
    DBScan_Pur_result = []

    for eps in DBSCAN_list["eps"]:
        max_silhouette = -2
        max_purity = -2

        for DBS in DBSCAN_list["min_sample"]:
            df_feature_pca = Data[clms]
            df_feature_pca = pca.fit_transform(df_feature_pca)
            df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])

            dbscan = DBSCAN(eps=eps, min_samples=DBS)
            label = dbscan.fit_predict(df_feature_pca)

            fig = px.scatter(
                df_feature_pca,
                x=df_feature_pca["PC1"],
                y=df_feature_pca["PC2"],
                color=label,
                title="DBSCAN"
            )
            #fig.show()

            # silhouette (range -1~1)
            try:
                current_silhouette = silhouette_score(Data, label, metric='euclidean')
            except:
                current_silhouette = -5

            if max_silhouette < current_silhouette:
                max_silhouette = current_silhouette

            # purity
            current_purity = purity_score(EM_Parameter, label)
            if max_purity < current_purity:
                max_purity = current_purity

        DBScan_Sil_result.append(max_silhouette)
        DBScan_Pur_result.append(max_purity)

    DBScan_Sil[preprocessing_name] = DBScan_Sil_result
    DBScan_pur[preprocessing_name] = DBScan_Pur_result

    # MeanShift
    print("Meanshift")
    Meanshift_Sil_result = []
    Meanshift_Pur_result = []

    for MS in MeanShift_list:
        meanShift = MeanShift(bandwidth=MS)
        label = meanShift.fit_predict(Data)

        df_feature_pca = Data[clms]
        df_feature_pca = pca.fit_transform(df_feature_pca)
        df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])


        label = meanShift.fit_predict(df_feature_pca)

        fig = px.scatter(
            df_feature_pca,
            x=df_feature_pca["PC1"],
            y=df_feature_pca["PC2"],
            color=label,
            title="MeanShift"
        )
        #fig.show()

        # silhouette (range -1~1)
        try:
            current_silhouette = silhouette_score(Data, label, metric='euclidean')
        except:
            current_silhouette = -1
        # silhouette (range -1~1)
        Meanshift_Sil_result.append(current_silhouette)

        # purity
        Meanshift_Pur_result.append(purity_score(EM_Parameter, label))

    Meanshift_Sil[preprocessing_name] = Meanshift_Sil_result
    Meanshift_pur[preprocessing_name] = Meanshift_Pur_result


# Test purity
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def ShowPlot(title, dict, x_list):
    for key, value in dict.items():
        plt.plot(x_list, value, label=key)

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.show()


def ResultPrint(x):
    result = {
        'sum': x.sum(),
        'count': x.count(),
        'mean': x.mean(),
        'variance': x.var()
    }
    return result


def FindBestResult(dict):
    key = None
    largest = 0
    for keys, item in dict.items():
        if max(item) > largest:
            largest = max(item)
            key = keys

    return key, largest


if __name__ == "__main__":
    main()