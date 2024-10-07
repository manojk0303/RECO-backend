from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
from pyngrok import ngrok
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datetime import datetime
print(flask.__version__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Replace this with your own API key from Google Cloud
YOUTUBE_API_KEY = 'YOUR_YOUTUBE_API_KEY'

# YouTube API service
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Load the tokenizer and model for sentiment analysis
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Convert ISO 8601 time to readable format (e.g., "3 days ago")
def convert_to_readable_time(iso_time):
    time_obj = datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%SZ")
    now = datetime.utcnow()
    time_diff = now - time_obj
    days = time_diff.days
    if days > 1:
        return f"{days} days ago"
    elif days == 1:
        return "1 day ago"
    else:
        seconds = time_diff.seconds
        if seconds >= 3600:
            return f"{seconds // 3600} hours ago"
        elif seconds >= 60:
            return f"{seconds // 60} minutes ago"
        else:
            return "Just now"

# Fetch comments from a YouTube video
def fetch_comments(video_id, token, page_token=None):
    youtube = build('youtube', 'v3', developerKey=token)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        pageToken=page_token,
        maxResults=50
    )
    response = request.execute()
    comments = []
    next_page_token = response.get('nextPageToken')

    for item in response['items']:
        comment_snippet = item['snippet']['topLevelComment']['snippet']
        comment = {
            'commentId': item['id'],
            'author': comment_snippet['authorDisplayName'],
            'text': comment_snippet['textOriginal'],
            'time': convert_to_readable_time(comment_snippet['publishedAt'])
        }
        comments.append(comment)

    return comments, next_page_token
#route for reply
@app.route('/generate_reply', methods=['POST'])
def generate_reply():
    data = request.json
    comment = data.get('comment')

    # Configure the Gemini API with your API key
    genai.configure(api_key="AIzaSyDPBpfrQsK-EJtwdP7PrfPmnkLW0zrSMAE")

    # Set up the generation configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Create the model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start a chat session
    chat_session = model.start_chat(
        history=[]
    )

    # Send a message to the model and get the response
    response = chat_session.send_message("Generate a reply for this comment (only one short and sweet reply alone): "+comment)
    reply =response.text
    print("gemini response: ")
    print(response.text)
    # if response.status_code == 200:
    print(reply)
    return jsonify({"reply": reply})
# Analyze sentiment in batches
def analyze_sentiment_batch(comments):
    if not comments:
        return []

    # Prepare the input for the model
    inputs = tokenizer([comment['text'] for comment in comments], return_tensors="pt", padding=True, truncation=True, max_length=256)

    # Perform predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Compute softmax probabilities and get the highest predicted sentiment
    probs = F.softmax(outputs.logits, dim=-1)
    sentiment_scores = torch.argmax(probs, dim=-1)

    # Map sentiment scores to labels
    labels = ["Negative", "Neutral", "Positive"]
    sentiment_results = [labels[score.item()] for score in sentiment_scores]

    # Attach the sentiment results to each comment
    for i, comment in enumerate(comments):
        comment['sentiment'] = sentiment_results[i]

    return comments

# Route for analyzing video comments
@app.route('/analyze', methods=['POST'])
def analyze_video():
    data = request.json
    video_id = data.get('videoId')
    page_token = data.get('pageToken')  # Accept pageToken for pagination
    token = data.get('token')

    # Fetch comments from YouTube API with pagination support
    fetched_comments, next_page_token = fetch_comments(video_id, token, page_token)

    # Analyze the sentiment of fetched comments
    comments_with_sentiment = analyze_sentiment_batch(fetched_comments)

    # Categorize comments by sentiment
    positive_comments = [comment for comment in comments_with_sentiment if comment['sentiment'] == 'Positive']
    negative_comments = [comment for comment in comments_with_sentiment if comment['sentiment'] == 'Negative']
    neutral_comments = [comment for comment in comments_with_sentiment if comment['sentiment'] == 'Neutral']

    response = {
        'Positive': positive_comments,
        'Negative': negative_comments,
        'Neutral': neutral_comments,
        'nextPageToken': next_page_token,  # Pass the nextPageToken to the client
        'allFetched': next_page_token is None  # Indicate if all comments have been fetched
    }

    return jsonify(response)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * Ngrok Tunnel URL: {public_url}")
    app.run(host='0.0.0.0', port=5000)
