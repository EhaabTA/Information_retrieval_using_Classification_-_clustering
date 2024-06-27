from django.core.management.base import BaseCommand
import os
from django.conf import settings
from ... import data_modules
from django.core.management.base import BaseCommand
import os
from django.conf import settings
import pandas as pd
import math
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from ... import data_modules
import json
import matplotlib.pyplot as plt


class Command(BaseCommand):
    help = 'Builds inverted index and IDF values'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting to build the inverted index and IDF values...")
        build_or_load_inverted_index()
        knn_classification()
        kmean_clustering()  # Implement k-means clustering (not shown)
        dump()
        save_image()

def remove_punctuations_and_numbers(text):
    text_no_punctuations = re.sub(r'[^\w\s]', '', text)
    text_no_punctuations_numbers = re.sub(r'\d', '', text_no_punctuations)
    return text_no_punctuations_numbers

def build_or_load_inverted_index():

    data_folder = os.path.join(settings.BASE_DIR, 'data')
    static_folder = os.path.join(data_folder, 'static')
    stopwords_file_path = os.path.join(data_folder, 'Stopword-List.txt')
    with open(stopwords_file_path) as file:
        stop_words = word_tokenize(file.read())

    modified_index = {}
    document_list = {}
    total_docs = 0
    ps = PorterStemmer()

    for filename in os.listdir(static_folder):
        file_path = os.path.join(static_folder, filename)
        if os.path.isfile(file_path):
            total_docs += 1
            with open(file_path, 'r', encoding='windows-1252') as file:
                word = file.read()
                text = remove_punctuations_and_numbers(word)
                words = word_tokenize(text)
                modified_index[total_docs] = [ps.stem(token.lower()) for token in words if token not in stop_words]
                document_list[total_docs] = (filename, len(modified_index[total_docs]))

    inverted_index = {}
    # key = term, value = [doc frequency,{Docid : frequency}]

    for filename, tokens in modified_index.items():
        for token in tokens:
            if token not in inverted_index.keys():
                inverted_index[token] = [1, {filename: 1}]
            else:
                dic = inverted_index[token]
                if filename in dic[1].keys():
                    dic[1][filename] += 1
                else:
                    dic[0] += 1
                    dic[1][filename] = 1

    def calculate_tfidf(idf, total_terms_in_doc, term_freq):
        tf = term_freq / total_terms_in_doc
        return tf * idf

    modified_index = None
    idf = {}
    tfidf_df_list = {}

    for term, values in inverted_index.items():
        df = values[0]  # Document frequency is stored at index 0 of values
        idf[term] = math.log(total_docs / (1 + df) + 1)
        tfidf = []
        docids = []

        for docid, term_freq in values[1].items():
            total_terms_in_doc = document_list[docid][1]  # Total terms in the document
            tfidf_val = calculate_tfidf(idf[term], total_terms_in_doc, term_freq)
            tfidf.append(tfidf_val)
            docids.append(docid)

        # Create a Series for the current term with document IDs as index and TF-IDF values as values
        tfidf_series = pd.Series(data=tfidf, index=docids)
        tfidf_df_list[term] = tfidf_series

    tfidf_df = pd.DataFrame(tfidf_df_list)
    tfidf_df.fillna(0, inplace=True)
            
    inverted_index = None
    tfidf_df_list = None
    data_modules.tfidf_df = tfidf_df
    data_modules.document_list = document_list
    data_modules.idf = idf
    data_modules.tfidf_df.to_json(os.path.join(data_folder, "tfidf_df"))
    

def knn_classification():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
    from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
    import pickle
    
    X = data_modules.tfidf_df
    y = list(data_modules.labels.values())
    
    if X.shape[0] != len(y):
        print("WARNING: Inconsistent data lengths! X has", X.shape[0], "rows, y has", len(y), "elements.")
        if X.shape[0] > len(y):  # Assuming missing label for the extra row in X
            print("Assuming missing label in y. Removing the last row from X.")
            X = X.iloc[:-1, :] 

    # Feature selection (optional, explore both chi2 and mutual_info_classif)
    selector = SelectKBest(chi2, k=150)  # Use chi2 or mutual_info_classif
    X = selector.fit_transform(X, y)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}  # Adjust parameters as needed
    knn_classifier = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_classifier, param_grid, scoring='accuracy')
    grid_search.fit(X, y)
    knn_classifier = grid_search.best_estimator_

    y_pred = knn_classifier.predict(X)

    data_modules.knn_precision_recall_f1 = classification_report(y, y_pred, output_dict=True, zero_division=1)
    data_modules.knn_accuracy = accuracy_score(y, y_pred)
    data_modules.knn_classifier = knn_classifier
    data_modules.knn_selector = selector
    print(data_modules.knn_accuracy, data_modules.knn_precision_recall_f1)

    path = os.path.join(settings.BASE_DIR, 'data','feature_selector.pkl')    
    with open(path, 'wb') as f:
        pickle.dump(selector, f)
        
    path = os.path.join(settings.BASE_DIR, 'data','knn_model.pkl')    
    with open(path, 'wb') as f:
        pickle.dump(knn_classifier, f)
    
def kmean_clustering():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import pickle
    import os
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    
    X = data_modules.tfidf_df
    y = data_modules.labels.values()
    if X.shape[0] != len(y):
        print("WARNING: Inconsistent data lengths! X has", X.shape[0], "rows, y has", len(y), "elements.")
        if X.shape[0] > len(y):  # Assuming missing label for the extra row in X
            print("Assuming missing label in y. Removing the last row from X.")
            X = X.iloc[:-1, :] 
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    # Saving the model
    path = os.path.join(settings.BASE_DIR, 'data', 'kmeans_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(kmeans, f)

    pred_labels = kmeans.labels_
    print(pred_labels)
    array = ["Explainable Artificial Intelligence","Heart Failure","Transformer Model","Time Series Forecasting", "Feature Selection"]
    true_labels = [array.index(row) for row in y]
    silhoute = silhouette_score(X, pred_labels)
    contingency = contingency_matrix(true_labels, pred_labels)
    purity = np.sum(np.max(contingency, axis=0)) / np.sum(contingency)
    contingency_matrix = contingency
    # Calculate TP (True Positives)
    TP = np.sum(np.diag(contingency_matrix))

    # Calculate TN (True Negatives)
    TN = np.sum(contingency_matrix) - TP

    # Calculate FP (False Positives)
    FP = np.sum(np.sum(contingency_matrix, axis=0)) - TP

    # Calculate FN (False Negatives)
    FN = np.sum(np.sum(contingency_matrix, axis=1)) - TP

    # Calculate Random Index
    random_index = (TP + TN) / (TP + TN + FP + FN)

    print("Random Index:", random_index)
    print("Silhoutte and Purity",silhoute,purity)
    data_modules.purity = purity
    data_modules.silhoute = silhoute
    data_modules.random_index = random_index
  

def dump():
    import json
    
    data = {
    "knn_precision_recall_f1": data_modules.knn_precision_recall_f1,
    "knn_accuracy": data_modules.knn_accuracy,
    "idf": data_modules.idf,
    "document_list": data_modules.document_list,
    }
    
    json_file_path = os.path.join(settings.BASE_DIR, 'data',"data.json")
    
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("JSON file created successfully:", json_file_path)
    
    
    
    
def save_image():
    
    json_file_path = os.path.join(settings.BASE_DIR, 'data',"data.json")

    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    
    image_dir = os.path.join('static', 'images')
    os.makedirs(image_dir, exist_ok=True)
      
    data = data["knn_precision_recall_f1"]      
    class_names = list(data.keys())[:-3]
    metrics = ['precision', 'recall', 'f1-score']

    # Plotting precision, recall, and F1-score for each class
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, [data[class_name][metric] for class_name in class_names], color='skyblue')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} for each Class')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        # Save the figure as an image
        image_path = os.path.join(image_dir, f'{metric}_bar_plot.png')
        plt.savefig(image_path)
        plt.close()
    
    averages = ['macro avg', 'weighted avg']
    for average in averages:
        plt.figure(figsize=(10, 4))
        plt.bar(metrics, [data[average][metric] for metric in metrics], color='lightgreen')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title(f'Averages ({average.capitalize()})')
        plt.ylim(0, 1)  # Set y-axis limit to 0-1
        plt.grid(axis='y')
        image_path = os.path.join(image_dir, f'{average}_bar_plot.png')
        plt.savefig(image_path)
        plt.close()

    metrics = ['Random Index', 'Silhouette Score', 'Purity']

    # Values for the metrics
    values = [data_modules.random_index, data_modules.silhoute, data_modules.purity]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'orange'])
    plt.title('Clustering Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Setting the y-axis limit to range from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    image_path = os.path.join(image_dir, f'clutering_bar_plot.png')
    plt.savefig(image_path)
    plt.close()
    
    
    global context
    
    context = {
        'precision_image_url': os.path.join(image_dir, 'precision_bar_plot.png'),
        'recall_image_url': os.path.join(image_dir, 'recall_bar_plot.png'),
        'f1_score_image_url': os.path.join(image_dir, 'f1_score_bar_plot.png'),
        'macro_image_url': os.path.join(image_dir, 'macro_avg_bar_plot.png'),
        'weighted_image_url': os.path.join(image_dir, 'weighted_avg_precision_bar_plot.png')
    }