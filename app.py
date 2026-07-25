from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from newspaper import Article
import feedparser
from urllib.parse import quote, urlparse
import os
import re
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer, util
import requests # Added for better error handling with URLs

# ---------------- APP CONFIGURATION ----------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- TRUSTED SOURCE REGISTRY ----------------
# Domain -> (Display name, homepage). Used to check whether a story is
# corroborated by outlets generally regarded as reliable, and to build the
# "Verified Sources" list shown on the result card.
TRUSTED_OUTLETS = {
    "reuters.com": ("Reuters", "https://www.reuters.com"),
    "apnews.com": ("AP News", "https://apnews.com"),
    "bbc.com": ("BBC", "https://www.bbc.com/news"),
    "bbc.co.uk": ("BBC", "https://www.bbc.co.uk/news"),
    "thehindu.com": ("The Hindu", "https://www.thehindu.com"),
    "indianexpress.com": ("Indian Express", "https://indianexpress.com"),
    "timesofindia.indiatimes.com": ("Times of India", "https://timesofindia.indiatimes.com"),
    "ndtv.com": ("NDTV", "https://www.ndtv.com"),
    "nytimes.com": ("The New York Times", "https://www.nytimes.com"),
    "cnn.com": ("CNN", "https://www.cnn.com"),
    "theguardian.com": ("The Guardian", "https://www.theguardian.com"),
    "pib.gov.in": ("PIB (Govt. of India)", "https://pib.gov.in"),
}

# Google News RSS gives us a human-readable publisher name in <source>
# (e.g. "BBC News", "Reuters") — not a resolvable domain, since its <link>
# is a redirect/proxy URL through news.google.com. So trusted-outlet
# detection has to also work on display-name substrings, not just domains.
TRUSTED_OUTLET_ALIASES = {
    "reuters": ("Reuters", "https://www.reuters.com"),
    "ap news": ("AP News", "https://apnews.com"),
    "associated press": ("AP News", "https://apnews.com"),
    "bbc": ("BBC", "https://www.bbc.com/news"),
    "the hindu": ("The Hindu", "https://www.thehindu.com"),
    "indian express": ("Indian Express", "https://indianexpress.com"),
    "times of india": ("Times of India", "https://timesofindia.indiatimes.com"),
    "ndtv": ("NDTV", "https://www.ndtv.com"),
    "new york times": ("The New York Times", "https://www.nytimes.com"),
    "nytimes": ("The New York Times", "https://www.nytimes.com"),
    "cnn": ("CNN", "https://www.cnn.com"),
    "the guardian": ("The Guardian", "https://www.theguardian.com"),
    "pib": ("PIB (Govt. of India)", "https://pib.gov.in"),
    "espn": ("ESPN", "https://www.espn.com"),
    "espncricinfo": ("ESPN Cricinfo", "https://www.espncricinfo.com"),
    "cricbuzz": ("Cricbuzz", "https://www.cricbuzz.com"),
    "press trust of india": ("Press Trust of India", "https://www.ptinews.com"),
}

def get_domain(url):
    """Extracts the bare registrable-ish domain (e.g. 'bbc.com') from a URL."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def match_trusted_outlet(value):
    """Returns (name, homepage) if the given domain, URL, or publisher
       display name (e.g. Google News RSS's <source> field, 'BBC News')
       belongs to a trusted outlet — else None."""
    if not value:
        return None

    # Strict domain matching first — this is the reliable path when we have
    # a real article URL (e.g. from URL-mode analysis).
    if "//" in value:
        domain = get_domain(value)
        for known_domain, info in TRUSTED_OUTLETS.items():
            if domain == known_domain or domain.endswith("." + known_domain):
                return info
        if domain.endswith(".gov") or domain.endswith(".gov.in"):
            return ("Government Source", value)

    # Fall back to matching the publisher's display name — this is the path
    # that actually fires for Google News related-article corroboration.
    value_lower = value.lower()
    for alias, info in sorted(TRUSTED_OUTLET_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in value_lower:
            return info

    return None

# ---------------- AI MODELS ----------------

# Fake News Classifier
# This model is generally good. We'll keep it as a primary classifier.
try:
    classifier = pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification"
    )
    print("Fake news classifier loaded successfully.")
except Exception as e:
    print(f"Error loading fake news classifier: {e}")
    classifier = None # Set to None if loading fails

# Semantic Similarity Model
# Used for finding related news articles.
try:
    similarity_model = SentenceTransformer(
        'all-MiniLM-L6-v2'
    )
    print("Sentence transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading sentence transformer model: {e}")
    similarity_model = None # Set to None if loading fails

# ---------------- TEXT CLEANING ----------------
def clean(text):
    """Cleans the input text by converting to lowercase, removing extra whitespace,
       and stripping leading/trailing spaces."""
    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespaces with a single space
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text) # Remove non-alphanumeric characters except punctuation
    return text

# ---------------- RULE ENGINE ----------------
def rule_engine(text):
    """Applies a set of rules to detect potentially fake news patterns.
       Returns (score, flags) where score is 0 (real) to 1 (fake) and
       flags is a list of plain-English reasons for the score."""
    if not text:
        return 0.0, []

    flags = []
    text_lower = text.lower() # Work with a lowercase version for rule matching

    suspicious_patterns = [
        "breaking shocking", "100% true", "share immediately", "viral claim",
        "secret leaked", "click here now", "forward this", "must watch",
        "you won't believe", "shocking truth", "exposed", "hidden agenda",
        "conspiracy", "hoax", "manipulated"
    ]

    impossible_pairs = [
        ("virat kohli", "silk smita"),
        ("dhoni", "silk smita"),
        ("elon musk", "ms dhoni"),
        ("president of india", "prime minister of china"), # Example of logical impossibility
        ("world is flat", "nasa confirmed")
    ]

    death_suicide_keywords = [
        "died", "dead", "death", "passed away", "killed", "funeral",
        "accident", "committed suicide", "suicide", "suicided",
        "hang himself", "hang herself", "took own life", "murdered"
    ]

    celebrities = [
        "ms dhoni", "dhoni", "virat kohli", "elon musk", "narendra modi",
        "salman khan", "shahrukh khan", "cristiano ronaldo", "lionel messi",
        "president biden", "president trump", "king charles"
    ]

    score = 0.0

    # 1. Suspicious phrasing
    matched_patterns = [p for p in suspicious_patterns if p in text_lower]
    for pattern in matched_patterns:
        score += 0.15 # Add a moderate score for each match
    if matched_patterns:
        flags.append("Contains sensational phrasing commonly seen in misleading posts (e.g. \"" + matched_patterns[0] + "\")")

    # 2. Impossible combinations of entities
    for celeb1, celeb2 in impossible_pairs:
        if celeb1 in text_lower and celeb2 in text_lower:
            score = max(score, 0.98) # Very high score if impossible entities are together
            flags.append(f"Mentions \"{celeb1.title()}\" and \"{celeb2.title()}\" together in a way that doesn't match any real, verifiable event")

    # 3. Celebrity death/suicide hoaxes
    for celeb in celebrities:
        if celeb in text_lower:
            for word in death_suicide_keywords:
                if word in text_lower:
                    score = max(score, 0.99) # Very high score for celebrity death hoaxes
                    flags.append(f"Claims a death/tragedy involving \"{celeb.title()}\" — this pattern matches common celebrity-death hoaxes")
                    break

    # 4. Sensationalism (e.g., excessive exclamation marks, ALL CAPS)
    if text.count('!') > 5 or text.isupper():
        score += 0.10
        flags.append("Excessive exclamation marks or ALL CAPS text, a common clickbait signal")

    # 5. URLs in text that are not from reputable domains (basic check)
    urls_in_text = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    reputable_domains = ["nytimes.com", "bbc.com", "reuters.com", "apnews.com", "cnn.com", "theguardian.com"]
    unreputable_found = False
    for url in urls_in_text:
        is_reputable = any(domain in url for domain in reputable_domains)
        if not is_reputable:
            score += 0.05 # Small penalty for unknown/suspicious domains
            unreputable_found = True
    if unreputable_found:
        flags.append("Links to a website outside our list of well-known, reputable news domains")

    return min(score, 1.0), flags # Ensure score doesn't exceed 1.0

# ---------------- SEMANTIC SIMILARITY FOR RELATED NEWS ----------------
def semantic_score(query, title):
    """Calculates the cosine similarity between a query and a title."""
    if not similarity_model:
        print("Semantic similarity model not loaded. Cannot calculate score.")
        return 0.0
    try:
        emb1 = similarity_model.encode(query, convert_to_tensor=True)
        emb2 = similarity_model.encode(title, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2)
        return float(score)
    except Exception as e:
        print(f"Error calculating semantic score: {e}")
        return 0.0

# ---------------- GET RELATED NEWS FROM GOOGLE NEWS ----------------
def get_related_news(query, max_results=8):
    """Fetches related news articles from Google News based on a query.
       Each result also carries the publisher name, a short summary, the
       published date, and whether the publisher is a trusted outlet."""
    if not query:
        return []

    # Encode the query for URL safety
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}"

    results = []
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            print(f"Feed parsing error: {feed.bozo_exception}")
            return []

        for entry in feed.entries:
            # Ensure entry.title and entry.link exist
            if not hasattr(entry, 'title') or not hasattr(entry, 'link'):
                continue

            # Calculate similarity to the original query
            score = semantic_score(query, entry.title)

            # Only add results that are semantically similar enough
            if score > 0.35: # Threshold can be adjusted
                source_obj = getattr(entry, "source", None)
                source_name = source_obj.get("title") if isinstance(source_obj, dict) else None
                if not source_name:
                    source_name = get_domain(entry.link) or "Unknown source"

                summary_raw = getattr(entry, "summary", "") or ""
                summary_clean = re.sub(r"<[^>]+>", "", summary_raw).strip()

                trusted = match_trusted_outlet(source_name) or match_trusted_outlet(entry.link)

                results.append({
                    "title": entry.title,
                    "link": entry.link,
                    "source": source_name,
                    "date": getattr(entry, "published", ""),
                    "summary": summary_clean[:180] + ("..." if len(summary_clean) > 180 else ""),
                    "score": round(score * 100, 2),
                    "trusted": bool(trusted)
                })

            if len(results) >= max_results:
                break

    except Exception as e:
        print(f"Error fetching related news: {e}")
        return []

    return results

# ---------------- ARTICLE EXTRACTION FROM URL ----------------
def extract_article(url):
    """Downloads and parses an article from a given URL."""
    if not url:
        return None

    try:
        # Use requests to check if the URL is accessible and get content type
        response = requests.get(url, timeout=10) # Add a timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Basic check for HTML content
        if 'text/html' not in response.headers.get('Content-Type', ''):
            print(f"URL is not HTML content: {url}")
            return None

        article = Article(url)
        article.download()
        article.parse()

        # Check if title and text were successfully extracted
        if not article.title or not article.text:
            print(f"Failed to extract title or text from URL: {url}")
            return None

        return {
            "title": article.title,
            "text": article.text
        }

    except requests.exceptions.RequestException as e:
        print(f"Request error for URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error extracting article from {url}: {e}")
        return None

# ---------------- OCR (Optical Character Recognition) ----------------
def extract_text_from_image(path):
    """Extracts text from an image file using Tesseract OCR."""
    if not path or not os.path.exists(path):
        return ""

    try:
        img = Image.open(path)
        # Ensure image is in RGB format for Tesseract
        if img.mode != "RGB":
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img)
        return text
    except FileNotFoundError:
        print(f"Image file not found at: {path}")
        return ""
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""

# ---------------- CORE ANALYSIS FUNCTION ----------------
def analyze_news(text, query_for_related_news="", title=""):
    """
    Analyzes the provided text to determine if it's likely real or fake news.
    Combines rule-based detection and an AI classifier.
    Returns a structured result dict (verdict, score, explanation, signals, sources, related_news).
    """
    empty_result = {
        "verdict": "NO TEXT FOUND", "score": 0, "related_news": [],
        "explanation": "No text was provided to analyze.",
        "positive_signals": [], "risk_signals": [], "sources": []
    }
    if not text:
        return empty_result

    cleaned_text = clean(text)

    if len(cleaned_text.split()) < 4: # Increased minimum word count for meaningful analysis
        empty_result["verdict"] = "TEXT TOO SHORT"
        empty_result["explanation"] = "The text is too short to analyze reliably. Please provide at least a full sentence."
        return empty_result

    # --- RULE ENGINE ANALYSIS ---
    rule_score, risk_signals = rule_engine(cleaned_text)

    # --- AI CLASSIFIER ANALYSIS ---
    ai_fake_score = 0.5 # Default to neutral if AI fails
    ai_ran = False
    ai_confidence = 0.0
    ai_label = None
    if classifier:
        try:
            # hamzab/roberta-fake-news-classification was fine-tuned on inputs
            # formatted EXACTLY as "<title>{title}<content>{content}<end>", using
            # the article's original casing. Feeding it plain, lowercased,
            # untagged text (as before) puts it far outside its training
            # distribution and makes its predictions unreliable/biased.
            pseudo_title = title.strip() if title and title.strip() else (
                text.strip().split(".")[0][:100] if text.strip() else "News"
            )
            content_for_model = text.strip()[:3000]  # tokenizer truncates to the model's real max length
            formatted_input = f"<title>{pseudo_title}<content>{content_for_model}<end>"

            result = classifier(formatted_input, truncation=True, max_length=512)[0]

            label = result["label"].upper()
            ai_confidence = result["score"]
            ai_ran = True

            # Handle either "LABEL_0"/"LABEL_1" or literal "FAKE"/"REAL" label schemes.
            if "FAKE" in label or label in ("LABEL_0", "0"):
                ai_fake_score = ai_confidence
                ai_label = "FAKE"
            elif "REAL" in label or "TRUE" in label or label in ("LABEL_1", "1"):
                ai_fake_score = 1 - ai_confidence
                ai_label = "REAL"
            else:
                print(f"Unrecognized classifier label scheme: {label}")
                ai_ran = False

        except Exception as e:
            print(f"AI Classifier error: {e}")
            # If AI fails, we rely more on the rule engine.
            pass
    else:
        print("AI classifier not loaded. Relying solely on rule engine.")


    # --- GET RELATED NEWS (moved up so corroboration can inform the score) ---
    # Use a relevant part of the text or title for searching related news.
    # If query_for_related_news is provided (e.g., from article title), use that.
    # Otherwise, use the beginning of the cleaned text.
    search_query = query_for_related_news if query_for_related_news else cleaned_text[:150]
    related_news = get_related_news(search_query)
    trusted_matches = [n for n in related_news if n.get("trusted")]
    trusted_outlet_names = []
    seen_outlet = set()
    for n in trusted_matches:
        if n["source"] not in seen_outlet:
            seen_outlet.add(n["source"])
            trusted_outlet_names.append(n["source"])
    trusted_count = len(trusted_outlet_names)

    # --- COMBINED SCORING ---
    # Higher weight to AI, but rule engine can override for extreme cases
    # Weights can be tuned based on observed performance.
    ai_weight = 0.75
    rule_weight = 0.25

    # If rule engine strongly suggests fake news, it should have a high impact.
    if rule_score >= 0.95:
        final_fake_score = 99.0 # Force a very high fake score
    else:
        final_fake_score = (
            (ai_fake_score * ai_weight) +
            (rule_score * rule_weight)
        ) * 100 # Convert to percentage

        # The AI classifier was fine-tuned on a static, dated dataset, so it has
        # no knowledge of anything that happened after its training cutoff and
        # can confidently mislabel recent-but-true stories as fake. Real-world
        # corroboration from trusted outlets is stronger evidence than a stale
        # classifier's guess, so let it pull the score toward REAL. It only
        # gets to *raise* suspicion a little when coverage exists but none of
        # it is from a trusted outlet — it never overrides a hard rule-engine
        # flag (handled above).
        if trusted_count >= 2:
            final_fake_score = min(final_fake_score, 25.0)
        elif trusted_count == 1:
            final_fake_score = max(0.0, final_fake_score - 25.0)
        elif related_news and trusted_count == 0:
            final_fake_score = min(100.0, final_fake_score + 10.0)

    # --- FINAL VERDICT ---
    # Thresholds can be adjusted based on desired sensitivity.
    if final_fake_score >= 65: # Adjusted threshold for fakeness
        verdict = "FAKE NEWS"
        score_value = round(final_fake_score, 2)
    else:
        # Calculate real score for display
        real_score_value = 100 - final_fake_score
        verdict = "REAL NEWS"
        score_value = round(real_score_value, 2)

    # --- BUILD POSITIVE SIGNALS ---
    positive_signals = []
    if ai_ran and ai_label == "REAL":
        positive_signals.append(f"Our AI classifier reads this as consistent with genuine reporting ({round(ai_confidence*100)}% confident)")
    if trusted_matches:
        positive_signals.append(
            f"{trusted_count} trusted outlet(s) — {', '.join(trusted_outlet_names[:4])} — are reporting closely related coverage"
        )
    if not risk_signals and not trusted_matches:
        positive_signals.append("No red-flag language patterns were detected in the text")

    # --- RISK SIGNALS (from rule engine) + AI disagreement ---
    if ai_ran and ai_label == "FAKE":
        risk_signals.append(f"Our AI classifier flags this as resembling fabricated stories ({round(ai_confidence*100)}% confident)")
    if not trusted_matches and related_news:
        risk_signals.append("None of the related coverage found comes from a widely trusted outlet")
    elif not related_news:
        risk_signals.append("No related coverage could be found for this story anywhere — treat this verdict as low-confidence")

    # --- VERIFIED SOURCES LIST ---
    sources = []
    seen_names = set()
    for n in trusted_matches:
        if n["source"] not in seen_names:
            seen_names.add(n["source"])
            homepage = n["link"]
            info = match_trusted_outlet(n["source"]) or match_trusted_outlet(n["link"])
            if info:
                homepage = info[1]
            sources.append({"name": n["source"], "url": homepage})

    # --- PLAIN-ENGLISH EXPLANATION ---
    verdict_word = "real" if verdict == "REAL NEWS" else "fake"
    explanation_parts = [f"Overall, this looks like {verdict_word} news with {score_value}% confidence."]
    if positive_signals:
        explanation_parts.append("Supporting this: " + "; ".join(positive_signals) + ".")
    if risk_signals:
        explanation_parts.append("Working against it: " + "; ".join(risk_signals) + ".")
    explanation = " ".join(explanation_parts)

    return {
        "verdict": verdict,
        "score": score_value,
        "related_news": related_news,
        "explanation": explanation,
        "positive_signals": positive_signals,
        "risk_signals": risk_signals,
        "sources": sources
    }

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    """Renders the main page."""
    return render_template("index.html")

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    """Analyzes text input directly."""
    try:
        data = request.get_json()
        text = data.get("text", "")

        result = analyze_news(text)

        # Provide a summary
        summary = clean(text)[:300] + ("..." if len(clean(text)) > 300 else "")

        return jsonify({
            "success": True,
            "verdict": result["verdict"],
            "score": result["score"],
            "title": "Text Analysis",
            "summary": summary,
            "related_news": result["related_news"],
            "explanation": result["explanation"],
            "positive_signals": result["positive_signals"],
            "risk_signals": result["risk_signals"],
            "sources": result["sources"]
        })

    except Exception as e:
        print(f"Error in /analyze_text: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/analyze_url", methods=["POST"])
def analyze_url():
    """Analyzes content from a given URL."""
    try:
        data = request.get_json()
        url = data.get("url")

        if not url:
            return jsonify({
                "success": False,
                "error": "No URL provided"
            })

        article_data = extract_article(url)

        if not article_data:
            return jsonify({
                "success": False,
                "error": "Unable to extract article from the provided URL. Please check the URL or try another."
            })

        title = article_data["title"]
        text = article_data["text"]

        result = analyze_news(text, query_for_related_news=title, title=title)

        # Provide a summary
        summary = clean(text)[:300] + ("..." if len(clean(text)) > 300 else "")

        return jsonify({
            "success": True,
            "verdict": result["verdict"],
            "score": result["score"],
            "title": title,
            "summary": summary,
            "related_news": result["related_news"],
            "explanation": result["explanation"],
            "positive_signals": result["positive_signals"],
            "risk_signals": result["risk_signals"],
            "sources": result["sources"]
        })

    except Exception as e:
        print(f"Error in /analyze_url: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """Analyzes text extracted from an uploaded image."""
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided in the request."
            })

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No selected file."
            })

        # Sanitize filename to prevent directory traversal
        filename = re.sub(r'[^\w\.\-]', '_', file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Ensure filename is unique if it already exists
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(path):
            filename = f"{base}_{counter}{ext}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            counter += 1

        file.save(path)

        # OCR to extract text
        extracted_text = extract_text_from_image(path)

        if not extracted_text or len(clean(extracted_text)) < 10: # Check if any meaningful text was extracted
            # Clean up the saved image file if no text is found
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error removing temporary image file {path}: {e}")
            return jsonify({
                "success": False,
                "error": "No readable text found in the image."
            })

        # Analyze the extracted text
        result = analyze_news(extracted_text)

        # Provide a summary
        summary = clean(extracted_text)[:300] + ("..." if len(clean(extracted_text)) > 300 else "")

        # Clean up the saved image file after analysis
        try:
            os.remove(path)
        except OSError as e:
            print(f"Error removing temporary image file {path}: {e}")

        return jsonify({
            "success": True,
            "verdict": result["verdict"],
            "score": result["score"],
            "title": "Image Analysis",
            "summary": summary,
            "related_news": result["related_news"],
            "explanation": result["explanation"],
            "positive_signals": result["positive_signals"],
            "risk_signals": result["risk_signals"],
            "sources": result["sources"]
        })

    except Exception as e:
        print(f"Error in /analyze_image: {e}")
        # Clean up the saved image file if an error occurs
        if 'path' in locals() and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as rm_e:
                print(f"Error removing temporary image file {path} after error: {rm_e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ---------------- RUN THE APP ----------------
if __name__ == "__main__":
    # It's recommended to use a more robust WSGI server (like Gunicorn) in production.
    # For local development, Flask's built-in server is fine.
    app.run(
        debug=True,         # Set to False in production
        host="0.0.0.0",     # Listen on all available network interfaces
        port=5000           # Port to run the server on
    )