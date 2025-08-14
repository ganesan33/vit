from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from typing import Optional, List
import praw, re, time, threading, os, requests
from datetime import datetime, timezone
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from geopy.geocoders import Nominatim
from geotext import GeoText

# --- Load environment variables ---
from dotenv import load_dotenv
load_dotenv()

# ========================
# Sentiment (NLTK VADER)
# ========================
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
SIA = SentimentIntensityAnalyzer()

# ========================
# FASTAPI APP SETUP
# ========================
app = FastAPI(title="Brand Monitoring API with Map + Summarization + Sentiment", version="8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ========================
# REDDIT CONFIG (from .env)
# ========================
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

# ========================
# SETTINGS
# ========================
MAX_WORKERS = 8
URGENT_KEYWORDS = ["bad", "hate", "worst", "cancel", "lawsuit", "boycott"]
URGENT_PATTERN = re.compile('|'.join(URGENT_KEYWORDS), re.IGNORECASE)

processed_ids = set()
karma_cache = {}
karma_lock = threading.Lock()
location_cache = {}
location_lock = threading.Lock()

geolocator = Nominatim(user_agent="reddit_brand_monitoring", timeout=10)

# ========================
# LOCATION MAP
# ========================
KNOWN_LOCATIONS = {
    "london": ("London, UK", 51.5074, -0.1278),
    "newyork": ("New York, USA", 40.7128, -74.0060),
    "nyc": ("New York, USA", 40.7128, -74.0060),
    "paris": ("Paris, France", 48.8566, 2.3522),
    "toronto": ("Toronto, Canada", 43.6532, -79.3832),
    "sanfrancisco": ("San Francisco, USA", 37.7749, -122.4194),
    "losangeles": ("Los Angeles, USA", 34.0522, -118.2437),
    "chicago": ("Chicago, USA", 41.8781, -87.6298),
    "berlin": ("Berlin, Germany", 52.52, 13.405),
    "sydney": ("Sydney, Australia", -33.8688, 151.2093),
    "melbourne": ("Melbourne, Australia", -37.8136, 144.9631),
}

# ========================
# GEMINI CONFIG (from .env)
# ========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# ========================
# REST OF YOUR EXISTING CODE
# ========================
# (Keep your scraping, utils, endpoints, etc. as they are)


# ========================
# REQUEST MODELS
# ========================
class ScrapeRequest(BaseModel):
    brands: List[str]
    subreddits: Optional[List[str]] = ["all"]
    time_filter: Optional[str] = "week"  # hour, day, week, month, year, all
    limit: Optional[int] = 15
    summarize: Optional[bool] = True  # Enable/disable Gemini summarization

# ========================
# UTILS
# ========================
@lru_cache(maxsize=1000)
def clean_text_fast(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'[\r\n\t]+', ' ', text.strip())

def get_author_karma_cached(author_obj, username):
    if not author_obj or username == "N/A":
        return "N/A"
    with karma_lock:
        if username in karma_cache:
            return karma_cache[username]
    try:
        total_karma = author_obj.link_karma + author_obj.comment_karma
        with karma_lock:
            karma_cache[username] = str(total_karma)
        return str(total_karma)
    except:
        with karma_lock:
            karma_cache[username] = "N/A"
        return "N/A"

@lru_cache(maxsize=1000)
def format_timestamp(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def detect_language_simple(text: str) -> str:
    if not text or len(text) < 10:
        return "en"
    english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a']
    count = sum(1 for w in english_words if f' {w} ' in f' {text.lower()} ')
    return "en" if count else "unknown"

def check_urgent_keywords(text: str) -> bool:
    return bool(URGENT_PATTERN.search(text))

def geocode_location_precise(location_name: str):
    if location_name.lower() in KNOWN_LOCATIONS:
        name, lat, lon = KNOWN_LOCATIONS[location_name.lower()]
        return lat, lon

    with location_lock:
        if location_name in location_cache:
            return location_cache[location_name]

    try:
        location = geolocator.geocode(location_name, exactly_one=True, language="en")
        if location:
            with location_lock:
                location_cache[location_name] = (location.latitude, location.longitude)
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Geocode error: {e}")
    return None, None

def detect_location(subreddit_name: str, texts: List[str]):
    possible_locations = set()

    if subreddit_name.lower() in KNOWN_LOCATIONS:
        loc_name, lat, lon = KNOWN_LOCATIONS[subreddit_name.lower()]
        return loc_name, lat, lon

    for text in texts:
        if text:
            places = GeoText(text)
            for city in places.cities:
                possible_locations.add(city)

    for loc in possible_locations:
        lat, lon = geocode_location_precise(loc)
        if lat and lon:
            return loc, lat, lon

    return None, None, None

def sentiment_of(text: str):
    """
    VADER sentiment: returns (compound, label)
    label in {'positive', 'neutral', 'negative'}
    """
    if not text:
        return 0.0, "neutral"
    scores = SIA.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label

# ========================
# REDDIT SCRAPING
# ========================
def process_submission(submission, keyword):
    try:
        if submission.id in processed_ids:
            return None
        processed_ids.add(submission.id)

        author_name = str(submission.author) if submission.author else "N/A"
        title_clean = clean_text_fast(submission.title)
        body_clean = clean_text_fast(submission.selftext or "")
        flair = submission.link_flair_text or ""

        # Combined text for NLP
        combined_text = f"{title_clean} {body_clean}".strip()

        loc_name, lat, lon = detect_location(
            submission.subreddit.display_name, [title_clean, body_clean, flair]
        )

        # Sentiment
        compound, senti_label = sentiment_of(combined_text)

        data = {
            "id": submission.id,
            "type": "post",
            "subreddit": submission.subreddit.display_name,
            "keyword": keyword,
            "title": title_clean[:250],
            "body": body_clean[:2500],
            "full_text_raw": (title_clean + " " + (submission.selftext or ""))[:5000],
            "language": detect_language_simple(combined_text),
            "author": author_name,
            "author_karma": get_author_karma_cached(submission.author, author_name),
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "flair": flair,
            "url": f"https://www.reddit.com{submission.permalink}",
            "created_utc": format_timestamp(submission.created_utc),
            "urgent_flag": check_urgent_keywords(combined_text),
            "lat": lat,
            "lon": lon,
            "location_name": loc_name,
            "sentiment_compound": compound,
            "sentiment_label": senti_label,
        }
        return data
    except Exception as e:
        print(f"Error processing submission {getattr(submission, 'id', 'unknown')}: {e}")
        return None

def search_reddit(reddit, subreddit_name, keyword, time_filter, limit):
    try:
        submissions = reddit.subreddit(subreddit_name).search(
            keyword, sort="new", time_filter=time_filter, limit=limit
        )
        results = []
        for sub in submissions:
            processed = process_submission(sub, keyword)
            if processed:
                results.append(processed)
        return results
    except Exception as e:
        print(f"Reddit search error in {subreddit_name} for {keyword}: {e}")
        return []

def create_reddit_instance():
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        timeout=15
    )

# ========================
# GEMINI SUMMARIZATION
# ========================
def generate_summary_gemini(text):
    if not GEMINI_API_KEY:
        return []
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GEMINI_API_KEY
    }
    prompt_with_instruction = (
        "Summarize the following social chatter into exactly 5 concise bullet points focused on brand sentiment, issues, and suggestions:\n\n"
        f"{text}"
    )
    data = {"contents": [{"parts": [{"text": prompt_with_instruction}]}]}
    try:
        r = requests.post(API_URL, headers=headers, json=data, timeout=30)
        r.raise_for_status()
        result = r.json()
        if "candidates" in result and result["candidates"]:
            text_block = result["candidates"][0]["content"]["parts"][0]["text"]
            summary_points = [pt.strip("-• ").strip() for pt in text_block.split("\n") if pt.strip()]
            return summary_points[:5]
        return []
    except Exception as e:
        return [f"[Summarization error: {str(e)}]"]

# ========================
# API ENDPOINTS
# ========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ScrapeRequestIn(BaseModel):
    brands: List[str]
    subreddits: Optional[List[str]] = ["all"]
    time_filter: Optional[str] = "week"
    limit: Optional[int] = 15
    summarize: Optional[bool] = True

@app.post("/scrape")
def scrape_brands(req: ScrapeRequestIn):
    processed_ids.clear()
    karma_cache.clear()
    data_rows = []

    search_keywords_map = {}
    for brand in req.brands:
        brand = brand.strip()
        if not brand:
            continue
        search_keywords_map[brand] = [
            brand,
            f"{brand} shoes",
            f"{brand} sneakers",
            f"{brand} reviews",
            f"{brand} complaints"
        ]

    if not search_keywords_map:
        return {"brands": [], "total_records": 0, "execution_time_sec": 0, "data": [], "summary": []}

    reddit = create_reddit_instance()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for sub in (req.subreddits or ["all"]):
            for brand, keywords in search_keywords_map.items():
                for kw in keywords:
                    futures.append(
                        executor.submit(search_reddit, reddit, sub, kw, req.time_filter, req.limit)
                    )
        for future in as_completed(futures):
            data_rows.extend(future.result())

    exec_time = round(time.time() - start_time, 2)

    summary = None
    if req.summarize and data_rows:
        combined_text = " ".join([post["full_text_raw"] for post in data_rows])
        summary = generate_summary_gemini(combined_text)

    # simple crisis signal = urgent posts or sudden cluster of negative
    urgent_count = sum(1 for d in data_rows if d.get("urgent_flag"))
    negative_count = sum(1 for d in data_rows if d.get("sentiment_label") == "negative")
    crisis_flag = (urgent_count >= 3) or (negative_count >= max(5, int(0.4 * max(1, len(data_rows)))))

    return {
        "brands": req.brands,
        "total_records": len(data_rows),
        "execution_time_sec": exec_time,
        "data": data_rows,
        "summary": summary or [],
        "signals": {
            "urgent_count": urgent_count,
            "negative_count": negative_count,
            "crisis_flag": crisis_flag
        }
    }

@app.get("/health")
def health():
    return {"ok": True}
