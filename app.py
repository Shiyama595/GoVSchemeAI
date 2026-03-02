# ---------------- IMPORTS ----------------
import os
import sqlite3
import random
import string
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import torch
# Change your flask import line to this:
from flask import Flask, render_template, request, redirect, session, jsonify, flash, url_for
import pandas as pd
# ADDED: jsonify to the imports so the chatbot can send data back to the browser
from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access the variables
ADMIN_USER = os.getenv("ADMIN_USERNAME")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD")

# ---------------- APP INIT ----------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FIXED: Removed the redundant 'os.path.join' wrapper on full paths to prevent path errors
dataset_path = r"C:\Users\New User\Downloads\government-scheme-recommendation-main\government-scheme-recommendation-main\dataset\Scheme_details.csv"
chatbot_dataset_path = r"C:\Users\New User\Downloads\government-scheme-recommendation-main\government-scheme-recommendation-main\dataset\updated_data.csv"
db_path = os.path.join(BASE_DIR, "database", "users.db")

os.makedirs(os.path.dirname(db_path), exist_ok=True)


# ---------------- LOAD DATASETS ----------------
scheme_df = pd.read_csv(dataset_path)

# Make all column names lowercase and clean
scheme_df.columns = [c.strip().lower() for c in scheme_df.columns]

# Load chatbot dataset
chatbot_df = pd.read_csv(chatbot_dataset_path)

# Convert numeric columns (based on your new dataset)
scheme_df["min_age"] = pd.to_numeric(scheme_df["min_age"], errors="coerce").fillna(0)
scheme_df["max_age"] = pd.to_numeric(scheme_df["max_age"], errors="coerce").fillna(120)
scheme_df["income"] = pd.to_numeric(scheme_df["income"], errors="coerce").fillna(float('inf'))

# Fill text columns
text_columns = ["gender", "category", "occupation", "area type", "disability"]

for col in text_columns:
    if col in scheme_df.columns:
        scheme_df[col] = scheme_df[col].fillna("all").astype(str).str.strip().str.lower()

print("Datasets Loaded ✅")

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user'
    )
    """)
    conn.commit()
    conn.close()

init_db()


# ---------------- CAPTCHA ----------------
def generate_captcha():
    captcha_text = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    session["captcha"] = captcha_text
    image = Image.new("RGB", (180, 70), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.text((30, 10), captcha_text, font=font, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        captcha_image = generate_captcha()
        return render_template("login.html", captcha_image=captcha_image)

    # 1. Get Form Data
    username = request.form.get("username")
    password = request.form.get("password")
    captcha_input = request.form.get("captcha")

    # 2. Verify Captcha
    if captcha_input.strip() != session.get("captcha"):
        flash("Captcha Incorrect ❌", "danger")
        return redirect(url_for('login'))

    # 3. PRIORITY CHECK: Is it the Admin? (From .env)
    if username == os.getenv("ADMIN_USERNAME") and password == os.getenv("ADMIN_PASSWORD"):
        session["user"] = username
        session["is_admin"] = True # Set a flag for the UI
        return redirect("/admin")

    # 4. DATABASE CHECK: Is it a regular user?
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = username
            session["is_admin"] = False
            return redirect("/dashboard")
        
    except Exception as e:
        print(f"Database Error: {e}")

    # 5. Fallback if everything fails
    flash("Invalid Credentials ❌", "danger")
    return redirect(url_for('login'))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template(
        "dashboard.html",
        genders=sorted(scheme_df["gender"].unique()),
        categories=sorted(scheme_df["category"].unique()),
        occupations=sorted(scheme_df["occupation"].unique()),
        area_types=sorted(scheme_df["area_type"].unique()),
        disabilities=sorted(scheme_df["disability"].unique()),
       
    )


# ---------------- CHATBOT LOGIC ----------------
# 1. Load a pre-trained Model (Lightweight and fast)
# 'all-MiniLM-L6-v2' is small but very accurate for text matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Prepare the Data
# We combine Scheme Name and Details into one string for better searching
chatbot_df['search_text'] = chatbot_df['scheme_name'] + " " + chatbot_df['details'].fillna("")
scheme_descriptions = chatbot_df['search_text'].tolist()

# 3. Pre-calculate Embeddings (Do this once at startup)
print("Encoding dataset for Semantic Search... ⏳")
corpus_embeddings = model.encode(scheme_descriptions, convert_to_tensor=True)
print("Chatbot Brain Ready! ✅")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("message", "").strip().lower()

    if not user_input:
        return jsonify({"reply": "Please type something!"})

    # 1. Encode User Query
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # 2. Get Top 3 Results (instead of just 1)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=3) # Get 3 best matches
    
    replies = []
    
    for score, idx in zip(top_results[0], top_results[1]):
        score = score.item()
        if score > 0.30:  # Threshold
            match = chatbot_df.iloc[idx.item()]
            name = match['scheme_name']
            # Create a small summary for each
            replies.append(f"• <b>{name}</b><br>")

    if replies:
        # If we found multiple, list them so the user can be specific
        header = "I found a few schemes that might match:<br><br>"
        main_match = chatbot_df.iloc[top_results[1][0].item()]
        
        reply = (header + "".join(replies) + 
         f"<br><b>Top Match Details:</b> {main_match['benefits']}")
    else:
        reply = "I couldn't find a match. Try being more specific, e.g., 'Health insurance for poor' or 'Farmer pension'."

    return jsonify({"reply": reply})
#------admin section-----------
from flask import render_template, session, redirect, url_for, flash
import pandas as pd

import sqlite3 # Add this to your imports at the top
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username").strip()
    password = request.form.get("password").strip()
    confirm_password = request.form.get("confirm_password").strip()

    # Basic validation
    if not username or not password:
        flash("All fields are required ❌", "danger")
        return redirect(url_for("register"))

    if password != confirm_password:
        flash("Passwords do not match ❌", "danger")
        return redirect(url_for("register"))

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if user already exists
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Username already exists ❌", "danger")
            conn.close()
            return redirect(url_for("register"))

        # Hash password before saving
        hashed_password = generate_password_hash(password)

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        conn.close()

        flash("Registration successful! Please login ✅", "success")
        return redirect(url_for("login"))

    except Exception as e:
        print("Registration Error:", e)
        flash("Something went wrong. Try again.", "danger")
        return redirect(url_for("register"))
@app.route("/admin")
def admin_dashboard():
    # 1. SECURITY: Check against .env variables
    if session.get('user') != ADMIN_USER:
        flash("Unauthorized Access! Admin privileges required.", "danger")
        return redirect(url_for('login')) 

    conn = None
    try:
        # 2. DATABASE: Connect using absolute path
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # FIX: Changed 'User' to 'users' to match your init_db() function
        cursor.execute("SELECT COUNT(*) FROM users") 
        actual_user_count = cursor.fetchone()[0]

        # 3. ANALYTICS: Prepare CSV data
        if scheme_df.empty:
            stats = {"total_schemes": 0, "total_users": actual_user_count }
            return render_template("admin_dashboard.html", stats=stats, all_schemes=[], 
                                   cat_labels=[], cat_values=[], occ_labels=[], occ_values=[])

        all_schemes = scheme_df.to_dict(orient='records')
        category_counts = scheme_df['category'].value_counts().to_dict()
        occ_counts = scheme_df['occupation'].value_counts().head(5).to_dict()
        
        # 4. OVERALL STATS
       

        stats = {
            "total_schemes": len(scheme_df),
            "total_users": actual_user_count
           
        }

        return render_template(
            "admin_dashboard.html", 
            all_schemes=all_schemes,
            cat_labels=list(category_counts.keys()), 
            cat_values=list(category_counts.values()),
            occ_labels=list(occ_counts.keys()),
            occ_values=list(occ_counts.values()),
            stats=stats
        )

    except Exception as e:
        print(f"🔥 Admin Error: {e}")
        flash(f"Error loading dashboard: {e}", "warning")
        return redirect(url_for('dashboard'))
    finally:
        if conn:
            conn.close()

# ---------------- RECOMMENDATION LOGIC ----------------
def check_eligibility(user, scheme):
    if not (scheme["min_age"] <= user["age"] <= scheme["max_age"]):
        return False
    if user["income"] > scheme["income"]:
        return False
    if scheme["gender"].lower() != "all" and scheme["gender"].lower() != user["gender"].lower():
        return False
    u_cat = user["category"].lower()
    s_cat = scheme["category"].lower()
    if s_cat != "all" and u_cat not in s_cat:
        return False
    if scheme["occupation"].lower() != "all" and scheme["occupation"].lower() != user["occupation"].lower():
        return False
    if scheme["area_type"].lower() != "all" and scheme["area_type"].lower() != user["area_type"].lower():
        return False
    if scheme["disability"].lower() == "yes" and user["disability"].lower() != "yes":
        return False
    return True


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user = {
            "age": int(request.form["age"]),
            "income": int(request.form["income"]),
            "gender": request.form["gender"],
            "category": request.form["category"],
            "occupation": request.form["occupation"],
            "area_type": request.form["area_type"],
            "disability": request.form["disability"]
        }
    except:
        return "Invalid Input ❌"

    eligible_schemes = []
    for _, scheme in scheme_df.iterrows():
        if check_eligibility(user, scheme):
            eligible_schemes.append(scheme)

    if not eligible_schemes:
        return render_template("result.html", schemes=[],user=user)

    result_df = pd.DataFrame(eligible_schemes)
    result_df = result_df.drop_duplicates(subset=["scheme_name"])
    result_df = result_df.sort_values(by="income")

    return render_template(
        "result.html",
        schemes=result_df.to_dict(orient="records"),
        user=user
    )

#-------logout--------
@app.route("/logout")
def logout():
    # Clear all data from the session
    session.clear() 
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('login')) # Redirect to login page

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

