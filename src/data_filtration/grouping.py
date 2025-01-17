import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class JournalCluster:
    def __init__(self, model_name='microsoft/deberta-large', cache_dir='./'):
        self.cache_dir = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir

        word_embedding_model = models.Transformer(model_name, cache_dir=cache_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], cache_folder=cache_dir)

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            self.journal_dict = json.load(f)
        return self.journal_dict

    def encode_texts(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings

    def find_optimal_clusters(self, embeddings, max_clusters=10):
        silhouette_scores = []
        max_clusters = min(max_clusters, len(embeddings) - 1)  # Ensure max_clusters is within valid range
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, labels)
            silhouette_scores.append((n_clusters, silhouette_avg))
        
        if silhouette_scores:
            # Find the number of clusters with the highest silhouette score
            best_num_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        else:
            best_num_clusters = 1  # Default to 1 cluster if no valid silhouette score is found
        
        return best_num_clusters

    def cluster_texts(self, embeddings, num_clusters):
        agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
        labels = agg_clustering.fit_predict(embeddings)
        return labels

    def create_dataframe(self):
        data = []
        for author_fullname, entries in tqdm(self.journal_dict.items(), desc="Processing authors"):
            author = entries[0]['author']
            ids = [entry['id'] for entry in entries]
            processed_texts = [entry['processed_text'] for entry in entries]
            
            if len(processed_texts) == 1:
                cluster_list = [[0]]
            else:
                embeddings = self.encode_texts(processed_texts)
                optimal_num_clusters = self.find_optimal_clusters(embeddings)
                clusters = self.cluster_texts(embeddings, optimal_num_clusters)
                
                # Organize clusters as lists of indices
                cluster_dict = {}
                for idx, cluster_id in enumerate(clusters):
                    if cluster_id not in cluster_dict:
                        cluster_dict[cluster_id] = []
                    cluster_dict[cluster_id].append(idx)
                
                cluster_list = list(cluster_dict.values())
            
            data.append({
                'author_fullname': author_fullname,
                'author': author,
                'id': ids,
                'processed_texts': processed_texts,
                'clusters': cluster_list  # List of clusters with indices
            })
        
        self.df = pd.DataFrame(data)
        return self.df

    def save_dataframe(self, file_path):
        self.df.to_json(file_path, orient='records', lines=True)

# Usage
