import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from groq import Groq
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Support Ticket System", page_icon="🤖")

st.title("🤖 AI-Powered Customer Support Ticket System")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV (ticket, category)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file with columns: ticket, category")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = {"ticket", "category"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain 'ticket' and 'category' columns.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Train Model
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ticket"].astype(str))

le = LabelEncoder()
y = le.fit_transform(df["category"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# -----------------------------
# Sentiment Analysis
# -----------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    return "Neutral"

# -----------------------------
# Priority Prediction
# -----------------------------
def get_priority(text):
    text = text.lower()

    high_keywords = [
        "urgent", "refund", "failed",
        "crash", "error", "critical"
    ]

    if any(word in text for word in high_keywords):
        return "High"
    elif len(text) > 50:
        return "Medium"
    else:
        return "Low"
from textblob import TextBlob
from groq import Groq
import matplotlib.pyplot as plt

GROQ_API_KEY = "groq_api_key"

client = Groq(api_key=GROQ_API_KEY)
# -----------------------------
# Groq Response
# -----------------------------
def generate_response(ticket, category):

    try:

        prompt = f"""
Customer Ticket:
{ticket}

Category:
{category}

Generate a professional customer support response.
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=150
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Groq Error: {e}"
# -----------------------------
# Prediction UI
# -----------------------------
st.subheader("Analyze New Ticket")

ticket = st.text_area("Enter Support Ticket")

if st.button("Analyze Ticket"):

    if ticket.strip() == "":
        st.warning("Please enter a ticket.")
    else:

        ticket_vector = vectorizer.transform([ticket])

        prediction = model.predict(ticket_vector)

        category = le.inverse_transform(prediction)[0]

        sentiment = get_sentiment(ticket)

        priority = get_priority(ticket)

        st.success(f"Category: {category}")
        st.info(f"Sentiment: {sentiment}")
        st.warning(f"Priority: {priority}")

        response = generate_response(ticket, category)

        st.subheader("AI Response")
        st.write(response)

# -----------------------------
# Dashboard
# -----------------------------
st.subheader("Dashboard")

counts = df["category"].value_counts()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values)
ax.set_title("Ticket Category Distribution")
plt.xticks(rotation=20)

st.pyplot(fig)

# -----------------------------
# Download Dataset
# -----------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Dataset",
    csv,
    "tickets.csv",
    "text/csv"
)
