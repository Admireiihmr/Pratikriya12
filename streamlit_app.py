import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import smtplib
import requests
import subprocess
import urllib.parse
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np
import ast
import re
import bcrypt
from pymongo import MongoClient




# ---------------------- Session State Init ----------------------
# Connect to MongoDB
client = MongoClient("mongodb+srv://admireiihmr:<db_password>@cluster0.9eczk3o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["Pratikriya"]
collection = db["users"]
csv_collection = db["csv"]
chart_collection = db["chart"]
email_collection = db["email"]

# Dummy user credentials
USER_CREDENTIALS = {
    "admin": "admin"

}


def hash_password(password):
    # Generate salt and hash the password
    salt = bcrypt.gensalt()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_pw


def store_user_credentials():
    for username, password in USER_CREDENTIALS.items():
        # Hash the password
        hashed_password = hash_password(password)

        # Create a document with hashed password
        user_doc = {
             "admin": username,
            "password": hashed_password
        }

        # Insert the document into MongoDB
        collection.insert_one(user_doc)


store_user_credentials()

print("User credentials (hashed) have been stored in MongoDB.")

def save_predictions_to_mongodb(df):
    records = df.to_dict(orient='records')    # Convert DataFrame to list of dicts
    csv_collection.insert_many(records)       # Insert into 'csv' collection
    #st.success("✅ Final prediction results saved into MongoDB!")  # Show success message


def save_chart_to_mongodb(chart_binary):
    chart_doc = {
        "filename": "sentiment_chart.png",
        "filedata": chart_binary
    }
    chart_collection.insert_one(chart_doc)
    st.success("✅ Diverging Sentiment Chart saved to MongoDB!")


def save_email_to_mongodb(email):
    email_doc = {
        "email_id": email
    }
    email_collection.insert_one(email_doc)
    #st.success("✅ Email ID saved to MongoDB!")



# ---------------------- Session State Init ----------------------


# Dummy user credentials
USER_CREDENTIALS = {
    "admin": "admin",
    "user": "admin"
}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False





if 'df' not in st.session_state:
    st.session_state.df1 = pd.DataFrame()


if 'users_db' not in st.session_state:
    st.session_state.users_db = {}

# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
#
# if 'auth_mode' not in st.session_state:
#     st.session_state.auth_mode = "login"

if 'reviews_loaded' not in st.session_state:
    st.session_state.reviews_loaded = False

if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

if 'email_done' not in st.session_state:
    st.session_state.email_done = False

b1 = False
b2 = False

# Keywords per aspect
aspect_keywords = {
    "doctor": ["dermatologist", "surgeon", "specialist", "doctors", "dentist", "endocrinology",
               "hospitalist", "neuro", "pulmonologist", "ctvs", "cardiac", "pediatrician", "intensivist",
               "md", "doctor", "physician", "immunologist", "dr", "practitioner", "consultant", "advisor",
               "registrar", "clinician", "resident", "urologist", "acupuncturist", "radiologist",
               "geriatrician", "pathologist", "anesthesiologist", "cardiologist", "cosmetologist",
               "endocrinologist", "gastroenterologist", "hepatologist", "gynecologist", "obstetrician",
               "oncologist", "neurologist", "ophthalmologist", "optometrist", "psychiatrist",
               "psychologist", "therapist", "rheumatologist", "nephrologist", "otolaryngologist",
               "hematologist", "toxicologist", "orthopedist", "podiatrist", "vaidyar", "rmo", "doc", "consultanting",
               "doctor", "consultation", "rude doctor", "good doctor", "bad doctor", "unprofessional",
               "treatment"],

    "staff": ["staff", "employee", "chemist", "nurse", "team", "staffs", "manager", "nursing",
              "technician", "guide", "employees", "providers", "workers", "professionals",
              "responders", "assistant", "housekeeping", "aayas", "reception", "admin",
              "receptionist", "matron", "groupd", "midwife", "pharmacist", "dietician",
              "nutritionist", "administrator", "executive", "clerk", "coordinator", "officer",
              "secretary", "engineer", "wardboy", "driver", "security", "personnel", "compounder",
              "perfusionist", "counselor", "guard", "nurses", "caregiver", "crew", "therapists",
              "facilitator", "faculty", "attendants", "rude", "nurse", "nurses", "icu", "behavior", "attitude", "staff",
              "rude", "impolite", "nurse", "bad service", "worst staff", "behavior", "irresponsible", "supervisors"],

    "cost": ["affordable", "expensive", "reasonable", "cost", "budget", "overpriced", "costeffective",
             "fare", "pricing", "fees", "payment", "insurance", "out-of-pocket", "costly", "premium",
             "luxurious", "billing", "pricey", "charges", "price", "free", "bills", "inexpensive",
             "cheap", "pocketfriendly", "value", "charge", "amount", "amounts", "bill", "expenditure",
             "finance", "penny", "coverage", "deductible", "negotiable", "overpaying", "clearance",
             "money", "economical", "rates", "minimum", "card", "cash", "costed", "priced", "rupees",
             "paid", "discount", "charging", "charged", "rs", "rps"],

    "waitingtime": ["delay", "stop", "wait", "remain", "long", "queue", "slow", "check-in", "hurried",
                    "crowded", "crowd", "busy", "late", "rush", "fast", "schedule", "line", "turnaround",
                    "waiting", "duration", "time", "speedy", "timely", "stagnation", "unhurried", "hurry",
                    "rapid", "pause", "scheduling", "sequence", "tardy", "hold", "backlog", "lag", "halt",
                    "stall", "quick", "postponed", "postpone", "response", "linger", "quickly", "hasty",
                    "minutes", "minute", "hour", "hours", "second", "seconds"],

    "cleanliness": ["clean", "aseptic", "antiseptic", "airy", "aromatic", "aesthtic", "decontamination",
                    "purity", "ablution", "neat", "tidy", "spotless", "infection", "sanitary", "hygienic",
                    "germ", "fumigation", "sterility", "autoclave", "oxidation", "chlorination", "fresh",
                    "immaculate", "sparkling", "pristine", "dust", "stains", "shiny", "linens", "bathroom",
                    "trash", "bins", "odor", "swept", "mopped", "spills", "windows", "walls", "polished",
                    "towels", "dustbin", "disposals", "cobwebs", "grime", "smell", "smudges", "disinfected",
                    "leakage", "mold", "surfaces", "sanitization", "dirt", "clear", "cluttered",
                    "uncluttered", "dirty", "scrubbed", "laundry", "foul", "odorless", "cleanliness",
                    "hygiene", "unhygienic", "rats", "room"],

    "facility": ["hospital", "office", "facility", "service", "hospital", "management", "accomodation", "people",
                 "experience", "convienent", "comfort", "air conditioning", "parking", "cafeteria",
                 "wheelchair", "elevators", "furniture", "lighting", "wi-fi", "seating", "signage",
                 "maintenance", "equipment", "spacious", "toiletries", "ventilation", "private", "shared",
                 "cabin", "garden", "area", "pharmacy", "lounge", "lab", "ward", "table",
                 "washbasins", "temperature", "lights", "storage", "disposal", "linen", "unit", "stay",
                 "care", "place", "healthcare", "hospitality", "comfortable", "atmosphere",
                 "overall", "organisation", "worth", "process", "clinic", "advanced", "department",
                 "quality", "speciality", "premises", "ambience", "environment", "privacy", "digital",
                 "everything", "infrastructure", "infra", "canteen", "corporate", "system", "centre",
                 "administration", "center", "standards", "clinics", "washroom", "toilet", "bathroom", "restroom",
                 "icu", "room access"]
}



def convert_to_T_format(review, aspect):
    review_lower = review.lower()
    for keyword in aspect_keywords.get(aspect, []):
        pattern = r'\b{}\b'.format(re.escape(keyword))
        if re.search(pattern, review_lower):
            return re.sub(pattern, "$T$", review_lower, count=1)
    return None

# Generate individual rows per aspect with $T$ formatting

def generate_T_format_rows(df, aspect_keywords):
    final_rows = []
    for _, row in df.iterrows():
        review = row['Reviews']

        # Safe evaluation of Aspects
        aspects_raw = row['Aspects']
        if isinstance(aspects_raw, str):
            try:
                aspects = ast.literal_eval(aspects_raw)
            except (ValueError, SyntaxError):
                aspects = []
        elif isinstance(aspects_raw, list):
            aspects = aspects_raw
        else:
            aspects = []

        # Safe evaluation of Sentiment
        sentiments_raw = row['Sentiment']
        if isinstance(sentiments_raw, str):
            try:
                sentiments = ast.literal_eval(sentiments_raw)
            except (ValueError, SyntaxError):
                sentiments = {}
        elif isinstance(sentiments_raw, dict):
            sentiments = sentiments_raw
        else:
            sentiments = {}

        for aspect in ["doctor", "staff", "cost", "cleanliness", "waitingtime", "facility"]:
            if aspect in sentiments and sentiments[aspect]:
                formatted_review = convert_to_T_format(review, aspect)
                if formatted_review:
                    final_rows.append({
                        "Reviews": review,
                        "Aspect": aspect,
                        "Formatted Review": formatted_review,
                        "Sentiment": sentiments[aspect]
                    })

    return pd.DataFrame(final_rows)

def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.jpg", width=300)

    st.title("🔐 Pratikriya HCQ - Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password.")


def scrape_google_reviews(url):


    hospital_name = url

    print("Var:", hospital_name)

    print(hospital_name)

    encoded_restaurant_name = urllib.parse.quote(hospital_name)

    print("Encode: ", str(encoded_restaurant_name))

    # Replace with your actual encoded restaurant name
    # url = f"https://api.app.outscraper.com/maps/reviews-v3?query={encoded_restaurant_name}&reviewsLimit=50&async=false"

    url = "https://api.app.outscraper.com/maps/reviews-v3?query={}&reviewsLimit=100&async=false".format(
        encoded_restaurant_name)
    # url = f"https://api.app.outscraper.com/maps/reviews-v3?query={encoded_restaurant_name}&reviewsLimit=50&async=false"
    api_key = "Z29vZ2xlLW9hdXRoMnwxMDg5ODYzMDg1ODI4NzIwODM1Nzh8ZGIwZGI2M2Q1Mg"

    curl_command = 'curl -X GET "{}" -H "X-API-KEY: {}"'.format(url, api_key)

    # Construct the CURL command
    # curl_command = f'curl -X GET "{url}" -H "X-API-KEY: {api_key}"'

    # Execute the CURL command and capture the response
    try:
        response_data = subprocess.check_output(curl_command, shell=True)
        print("Response data captured successfully.")
        print(response_data)
        # Now you can use response_data as your variable containing the JSON response.
    except subprocess.CalledProcessError as e:
        print("Error")

    json_response = json.loads(response_data.decode('utf-8'))

    num_reviews = len(json_response['data'][0]['reviews_data'])

    reviews_list = []
    for i in range(num_reviews):
        a1 = json_response['data'][0]['reviews_data'][i]['author_title']
        a2 = json_response['data'][0]['reviews_data'][i]['review_text']

        # append a1 and a2 to a list
        reviews_list.append((a1, a2))

    df = pd.DataFrame(reviews_list, columns=['Author', 'Reviews'])
    df['Hospital Name'] = hospital_name
    return df

from email.message import EmailMessage
# ---------------------- Main App ----------------------
def send_email_with_chart(to_email, chart_buffer=None):
    from_email = 'akhilaa1326@gmail.com'
    password = 'vhjb rweg mzht sdkr'
    subject = "Sentiment Analysis Chart"
    body = "Attached is your sentiment chart."

    with open("sentiment_chart.png", "rb") as f:
        chart_data = f.read()

    # Create the email message
    msg = EmailMessage()
    msg['Subject'] = '📊 Sentiment Analysis Chart'
    msg['From'] = from_email # Replace with your email
    msg['To'] = to_email
    msg.set_content("Attached is the sentiment chart you requested.")

    # Attach the PNG file
    msg.add_attachment(chart_data, maintype='image', subtype='png', filename='sentiment_chart.png')

    try:
        # Send the email using Gmail SMTP (you can change this for other providers)
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(from_email, password)  # Use an App Password!
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False
@st.cache_resource
def load_aspect_keywords():
    with open("aspect_keywords.json", "r") as f:
        return json.load(f)

def extract_aspects_with_keywords(review, aspect_keywords):
    matched_aspects = []
    matched_keywords = []
    review_lower = review.lower()
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            pattern = r'\b{}\b'.format(re.escape(keyword))
            if re.search(pattern, review_lower):
                matched_aspects.append(aspect)
                matched_keywords.append(keyword)
    return matched_aspects, matched_keywords


def preprocess1(df1):
    aspects = ["doctor", "staff", "cost", "cleanliness", "waitingtime", "facility"]
    aspect_df = df1['Sentiment'].apply(pd.Series)
    df = pd.concat([df1, aspect_df], axis=1)

    def sentiment_to_score(sentiment):
        if isinstance(sentiment, str):
            if sentiment == 'Positive':
                return 1
            elif sentiment == 'Negative':
                return -1
            elif sentiment == 'Negative':
                return 0
            else:
                return np.nan
        return 0

    for aspect in aspects:
        score_col = aspect + ' (Vector)'
        df[score_col] = df.apply(lambda row: sentiment_to_score(row[aspect]), axis=1)

    percentage_results = []
    for aspect in aspects:
        # print(aspect)
        aspect_score_col = aspect + ' (Vector)'

        if aspect_score_col in df.columns:
            #print(aspect_score_col)
            filtered_df = df[df[aspect_score_col].notna()]
            # print(filtered_df.shape)
            is_empty = filtered_df[aspect_score_col].isna().all()
            if not is_empty:
                grouped_counts = filtered_df[aspect_score_col].value_counts().sort_index()

                total_count = grouped_counts.sum()

                grouped_percentages = (grouped_counts / total_count) * 100

                aspect_percentage = {'Aspect': aspect}
                aspect_percentage['Positive'] = grouped_percentages.get(1, 0)
                aspect_percentage['Neutral'] = grouped_percentages.get(0, 0)
                aspect_percentage['Negative'] = grouped_percentages.get(-1, 0)

                percentage_results.append(aspect_percentage)

            else:
                print(f"{aspect_score_col} column not found in the DataFrame.")
                # Initialize dictionary for current aspect with 0 values
                aspect_percentage = {'Aspect': aspect, 'Positive': 0, 'Neutral': 0, 'Negative': 0}
                percentage_results.append(aspect_percentage)
    df_percentage = pd.DataFrame(percentage_results)

    def get_max_category(row):
        max_value = row[1:].astype(float).max()
        max_category = row[1:].astype(float).idxmax()
        return f"-{max_value}" if max_category == "Negative" else str(max_value)

    df_percentage["Sentiment"] = df_percentage.apply(get_max_category, axis=1)

    return df_percentage

def extract_aspects(review, aspect_keywords):
    matched = []
    review_lower = review.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(re.search(r'\b{}\b'.format(re.escape(keyword)), review_lower) for keyword in keywords):
            matched.append(aspect)
    return matched



def calculate_ttr(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

def jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    intersection = a.intersection(b)
    union = a.union(b)
    return len(intersection) / len(union) if union else 0

def calculate_batch_ttr_jaccard(reviews):
    total_ttr = sum(calculate_ttr(review) for review in reviews)
    avg_ttr = round(total_ttr / len(reviews), 3) if reviews else 0

    pairwise_jaccard = []
    for i in range(len(reviews)):
        for j in range(i + 1, len(reviews)):
            sim = jaccard_sim(reviews[i], reviews[j])
            pairwise_jaccard.append(sim)

    avg_jaccard = round(sum(pairwise_jaccard) / len(pairwise_jaccard), 3) if pairwise_jaccard else 0
    return avg_ttr, avg_jaccard

def main_app():
    st.title("📊 Pratikriya HCQ - Sentiment Prediction")

    chart_buffer = None
    aspect_keywords = load_aspect_keywords()

    option = st.radio("Choose input method", ["📤 Upload File", "🏥 Enter Hospital Name"])

    reviews = []

    if option == "📤 Upload File":
        #st.session_state.prediction_done = False
        st.session_state.email_done = False
        st.session_state.reviews_loaded = False
        uploaded_file = st.file_uploader("Upload a CSV or Excel file with a 'Reviews' column", type=["csv", "xlsx"])

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                # return

            if 'Reviews' not in df.columns:
                st.error("The uploaded file must contain a 'Reviews' column.")
                return

                st.session_state.df1 = df.copy()
                reviews = df['Reviews'].dropna().astype(str).tolist()
                st.session_state.reviews_loaded = True
                st.success(f"{len(reviews)} reviews loaded successfully!")
            else:
                reviews = df['Reviews'].astype(str).tolist()
                st.session_state.file_loaded = True
                # return
            reviews = df['Reviews'].astype(str).tolist()
            st.session_state.file_loaded = True


    # Option UI
    if option == "🏥 Enter Hospital Name":
        st.session_state.email_done = False
        st.session_state.file_loaded = False
        hospital_name = st.text_input("Enter the hospital name for scraping reviews")

        if st.button("Scrape Reviews"):
            if not hospital_name:
                st.warning("⚠️ Please enter a hospital name.")
            else:
                df = scrape_google_reviews(hospital_name)
                df.to_csv("ScrappedReviews.csv")

                if df.shape[0] != 0:
                    st.session_state.reviews_loaded = True
                    st.success("✅ Reviews loaded successfully!")


                else:
                    st.session_state.reviews_loaded = False
                    st.warning("⚠️ Please enter a valid hospital name.")

        #     if not hospital_name:
        #         st.warning("⚠️ Please enter a hospital name.")
        #     else:
        #         #df = scrape_google_reviews(hospital_name)
        #
        #df = pd.read_csv("Pratikriya.csv")

    # ---------- Run sentiment prediction ----------
    if st.session_state.reviews_loaded:
        st.write("🔍 Ready to Predict Sentiments")
        if st.button("Predict Sentiments from reviews"):
            with st.spinner("🔍 Processing reviews..."):
                df = pd.read_csv("ScrappedReviews.csv")
                reviews = df['Reviews'].dropna().astype(str).tolist()

                # 🔍 Batch-Level Spam Detection
                st.subheader("📉 Batch-Level Spam Detection")
                avg_ttr, avg_jaccard = calculate_batch_ttr_jaccard(reviews)
                st.markdown(f"**Average Type-Token Ratio (TTR):** `{avg_ttr}`")
                st.markdown(f"**Average Jaccard Similarity:** `{avg_jaccard}`")

                # Threshold check
                TTR_THRESHOLD = 0.2
                JACCARD_THRESHOLD = 0.8

                if avg_ttr <= TTR_THRESHOLD and avg_jaccard >= JACCARD_THRESHOLD:
                    st.error("🚫 Reviews flagged as spam (Low TTR + High Jaccard). Skipping prediction.")
                    return
                else:
                    st.success("✅ Passed spam check. Proceeding to sentiment prediction...")

                # Extract aspects
                df[['Aspects', 'Matched Keywords']] = df["Reviews"].apply(
                    lambda x: pd.Series(extract_aspects_with_keywords(x, aspect_keywords))
                )

                # API call
                response = requests.post("http://127.0.0.1:5003/predict", json={"reviews": reviews})
                if response.status_code == 200:
                    sentiments = response.json()["results"]
                    df["Sentiment"] = sentiments
                    st.success("✅ Sentiment prediction completed!")
                else:
                    st.error("❌ API request failed. Check FastAPI server.")
                    return

            # ---------- Chart Visualization ----------
            if not df.empty:
                st.subheader("📈 Diverging Sentiment Chart")
                dfEdit = preprocess1(df)
                parameters = list(dfEdit['Aspect'])
                sentiment_percentages = [float(s) for s in dfEdit['Sentiment']]
                colors = ['green' if s >= 0 else 'red' for s in sentiment_percentages]

                fig, ax = plt.subplots(figsize=(18, 8))
                bars = ax.barh(parameters, sentiment_percentages, color=colors)

                for bar, sentiment in zip(bars, sentiment_percentages):
                    ax.text(
                        bar.get_width() + (0.01 if sentiment >= 0 else -0.01),
                        bar.get_y() + bar.get_height() / 2,
                        f'{sentiment:.1f}',
                        va='center',
                        ha='left' if sentiment >= 0 else 'right',
                        fontsize=10,
                        color='black'
                    )

                ax.set_xlabel('Sentiment Percentage')
                ax.set_ylabel('Aspect')
                ax.set_title('Sentiment Percentage for Different Aspects')
                fig.savefig("sentiment_chart.png", format='png', bbox_inches='tight')
                chart_buffer = io.BytesIO()
                fig.savefig(chart_buffer, format="png")
                chart_buffer.seek(0)

                st.session_state.prediction_done = True
                st.session_state.email_done = True
                st.session_state.chart_buffer = chart_buffer.getvalue()

                st.pyplot(fig)

                # 📥 Download Chart
                st.download_button(
                    label="📥 Download Chart as PNG",
                    data=st.session_state.chart_buffer,
                    file_name="sentiment_chart.png",
                    mime="image/png"
                )

                # 📥 Download Results
                st.markdown("### 📥 Download Results")
                formatted_df = generate_T_format_rows(df, aspect_keywords)
                result_csv = formatted_df[["Reviews", "Aspect", "Formatted Review", "Sentiment"]]
                csv_buffer = io.StringIO()
                result_csv.to_csv(csv_buffer, index=False)
                st.download_button("Download CSV", csv_buffer.getvalue(), file_name="sentiment_results.csv",
                                   mime="text/csv")

            st.session_state.reviews_loaded = False

    # ---------- Run sentiment prediction ----------
    if st.session_state.file_loaded:
        st.write("🔍 Ready to Predict Sentiments")
        if st.button("Predict Sentiments from File"):
            with st.spinner("🔍 Processing uploaded reviews..."):

                # Load and validate reviews
                reviews = df['Reviews'].dropna().astype(str).tolist()

                # 📉 Batch-Level Spam Detection
                st.subheader("📉 Batch-Level Spam Detection")
                avg_ttr, avg_jaccard = calculate_batch_ttr_jaccard(reviews)
                st.markdown(f"**Average Type-Token Ratio (TTR):** `{avg_ttr}`")
                st.markdown(f"**Average Jaccard Similarity:** `{avg_jaccard}`")

                TTR_THRESHOLD = 0.2
                JACCARD_THRESHOLD = 0.8

                if avg_ttr <= TTR_THRESHOLD and avg_jaccard >= JACCARD_THRESHOLD:
                    st.error("🚫 Reviews flagged as spam (Low TTR + High Jaccard). Skipping prediction.")
                    return
                else:
                    st.success("✅ Passed spam check. Proceeding to sentiment prediction...")

                # Extract aspects
                df[['Aspects', 'Matched Keywords']] = df["Reviews"].apply(
                    lambda x: pd.Series(extract_aspects_with_keywords(x, aspect_keywords))
                )

                # Call prediction API
                response = requests.post("http://127.0.0.1:5003/predict", json={"reviews": reviews})
                if response.status_code == 200:
                    sentiments = response.json()["results"]
                    df["Sentiment"] = sentiments
                    st.success("✅ Sentiment prediction completed!")
                else:
                    st.error("❌ API request failed. Check FastAPI server.")
                    return

            # ---------- Chart Visualization ----------
            if not df.empty:
                st.subheader("📈 Diverging Sentiment Chart")
                dfEdit = preprocess1(df)
                parameters = list(dfEdit['Aspect'])
                sentiment_percentages = [float(s) for s in dfEdit['Sentiment']]
                colors = ['green' if s >= 0 else 'red' for s in sentiment_percentages]

                fig, ax = plt.subplots(figsize=(18, 8))
                bars = ax.barh(parameters, sentiment_percentages, color=colors)

                for bar, sentiment in zip(bars, sentiment_percentages):
                    ax.text(
                        bar.get_width() + (0.01 if sentiment >= 0 else -0.01),
                        bar.get_y() + bar.get_height() / 2,
                        f'{sentiment:.1f}',
                        va='center',
                        ha='left' if sentiment >= 0 else 'right',
                        fontsize=10,
                        color='black'
                    )

                ax.set_xlabel('Sentiment Percentage')
                ax.set_ylabel('Aspect')
                ax.set_title('Sentiment Percentage for Different Aspects')

                fig.savefig("sentiment_chart.png", format='png', bbox_inches='tight')

                chart_buffer = io.BytesIO()
                fig.savefig(chart_buffer, format="png")
                chart_buffer.seek(0)

                st.session_state.prediction_done = True
                st.session_state.email_done = True
                st.session_state.chart_buffer = chart_buffer.getvalue()

                st.pyplot(fig)

                # ---------- Download Chart ----------
                st.download_button(
                    label="📥 Download Chart as PNG",
                    data=st.session_state.chart_buffer,
                    file_name="sentiment_chart.png",
                    mime="image/png"
                )

                # ---------- Download Results ----------
                st.markdown("### 📥 Download Results")
                formatted_df = generate_T_format_rows(df, aspect_keywords)
                result_csv = formatted_df[["Reviews", "Aspect", "Formatted Review", "Sentiment"]]
                csv_buffer = io.StringIO()
                result_csv.to_csv(csv_buffer, index=False)
                st.download_button("Download CSV", csv_buffer.getvalue(), file_name="sentiment_results.csv",
                                   mime="text/csv")

            st.session_state.file_loaded = False

            # st.markdown("### 📧 Send chart via email")
            # with st.form("email_form"):
            #     to_email = st.text_input("Recipient Email", placeholder="Enter email address")
            #     send_it = st.form_submit_button("Send Email")
            #     print(send_it)
            #     if send_it:
            #         if to_email:
            #             if send_email_with_chart(to_email):
            #                 st.success("✅ Email sent successfully.")
            #
            #             else:
            #                 st.error("❌ Email failed.")
            #         else:
            #             st.warning("⚠️ Please enter a valid email.")


    # #---------- Email Functionality ----------
    if st.session_state.get("prediction_done", True):
        st.markdown("### 📧 Send chart via email")
        with st.form("email_form"):
            to_email = st.text_input("Recipient Email", placeholder="Enter email address")
            # send_it = st.form_submit_button("Send Email")
            # print(send_it)
            if st.form_submit_button("Send Email"):
                if to_email:
                    if send_email_with_chart(to_email):
                        st.success("✅ Email sent successfully.")

                    else:
                        st.error("❌ Email failed.")
                else:
                    st.warning("⚠️ Please enter a valid email.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()


# App logic
if not st.session_state.logged_in:
    login()
else:
    main_app()
