# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import jsonlines

def process_clusters(input_file, output_file):
    with jsonlines.open(input_file, 'r') as infile, jsonlines.open(output_file, 'w') as outfile:
        for data in infile:
            clusters = data.get('clusters', [])
            if not clusters:
                print(f"No clusters found for author: {data.get('author_fullname', 'Unknown')}")
                continue

            max_cluster = max(clusters, key=len)
            
            filtered_ids = [data['id'][i] for i in max_cluster]
            filtered_texts = [data['processed_texts'][i] for i in max_cluster]
            
            filtered_data = {
                'author_fullname': data['author_fullname'],
                'author': data['author'],
                'id': filtered_ids,
                'processed_texts': filtered_texts
            }
            
            outfile.write(filtered_data)


# Specify the input and output file paths
input_file = './output.jsonl'
output_file = './filtered.jsonl'

# Process the clusters and create the new JSONL file
process_clusters(input_file, output_file)