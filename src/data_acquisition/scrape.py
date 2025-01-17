import requests
import time
from tqdm.auto import tqdm

# Read the word_list(query list) from a file
with open('words_list.txt', 'r') as file:
    words_list = [line.strip() for line in file]
print(words_list)

queries = words_list
# URL template
# base_url = "https://api.pullpush.io/reddit/search/submission/?q={query}&subreddit=DiaryOfARedditor"
base_url = "https://api.pullpush.io/reddit/search/submission/?q={query}&subreddit=Journaling"

# Set to store unique submission data
unique_submissions = set()

# Dictionary to hold the data
submissions_data = []

# Loop through each query and fetch data
for query in tqdm(queries, desc="Processig Queries"):
    try:
        response = requests.get(base_url.format(query=query))
        if response.status_code == 200:
            submissions = response.json()
            submissions = submissions['data']
            for submission in submissions:
                submission_id = submission.get('id', '')
                if submission_id and submission_id not in unique_submissions:
                    unique_submissions.add(submission_id)
                    # Store required submission data
                    submission_data = {
                        'author': submission.get('author', ''),
                        'author_fullname': submission.get('author_fullname', ''),
                        'id': submission_id,
                        'selftext': submission.get('selftext', ''),
                        'created_utc': submission.get('created_utc', ''),
                        'title': submission.get('title', ''),
                        'url': submission.get('url', '')
                    }
                    submissions_data.append(submission_data)
    except Exception as e:
          print(f"Error fetching data for query: {query}")
          continue

    # Wait for 1 second before making the next request to avoid hitting rate limits
    time.sleep(1)

# At this point, `submissions_data` contains all unique submissions data.
print(f"Total unique submissions fetched: {len(submissions_data)}")

# Save raw data to CSV
df.to_csv('submissions.csv')