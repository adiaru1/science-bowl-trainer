import os
import json
import random
import string
import glob
import time
import re
import traceback
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader

app = Flask(__name__, template_folder=".")

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=15000,
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)

FAST_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

def load_rag_chunks():
    os.makedirs("packets", exist_ok=True)
    chunks = []
    pdf_files = glob.glob("packets/*.pdf")
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text() or ""
                parts = text.split("TOSS-UP")
                if len(parts) > 1:
                    for part in parts[1:]:
                        subparts = part.split("BONUS")
                        tossup_text = subparts[0].strip()
                        bonus_text = subparts[1].strip() if len(subparts) > 1 else ""
                        if len(tossup_text) > 40:
                            chunks.append({
                                "id": len(chunks),
                                "tossup": tossup_text[:1400],
                                "bonus": bonus_text[:1400] if bonus_text else tossup_text[:1400]
                            })
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
    random.shuffle(chunks)
    print(f"=== RAG LOADED: {len(chunks)} question blocks from {len(pdf_files)} PDFs ===")
    return chunks

RAG_CHUNKS = load_rag_chunks()
USED_CHUNK_IDS = set()
ASKED_HISTORY = []
CONTESTS = {}

SUBTOPICS = {
    "Biology": ["Cell Biology", "Genetics & Evolution", "Human Anatomy", "Botany", "Ecology", "Microbiology", "Biochemistry"],
    "Chemistry": ["Stoichiometry", "Thermochemistry", "Organic Chemistry", "Acids & Bases", "Electrochemistry", "Periodic Trends", "Kinetics"],
    "Physics": ["Kinematics & Mechanics", "Electricity & Magnetism", "Optics & Waves", "Thermodynamics", "Quantum Physics", "Fluid Mechanics"],
    "Mathematics": ["Algebra", "Geometry & Trigonometry", "Combinatorics & Probability", "Number Theory", "Precalculus", "Calculus"],
    "Earth and Space": ["Astronomy", "Planetary Science", "Plate Tectonics", "Meteorology", "Oceanography", "Mineralogy"],
    "Energy": ["Renewable Energy", "Nuclear Fission & Fusion", "Batteries & Storage", "Thermodynamics", "Power Grid", "DOE Labs Research"]
}

class QuestionData(BaseModel):
    category: str
    q_type: str
    q_format: str
    question: str
    answer: str

class GradeData(BaseModel):
    correct: bool
    explanation: str

def call_gemini_fast(prompt, schema, temp):
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in Render Environment Variables!")
    last_error = None
    for model_name in FAST_MODELS:
        clean_model = model_name.strip()
        if clean_model.startswith("models/"):
            clean_model = clean_model.replace("models/", "", 1)

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=clean_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temp,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                last_error = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(1)
                    continue
                break
    raise last_error

def get_random_chunk(category, q_type, chunk_id=None):
    if not RAG_CHUNKS:
        return None, None
    if chunk_id is not None:
        match = next((c for c in RAG_CHUNKS if c["id"] == chunk_id), None)
        if match:
            text = match["bonus"] if q_type.lower() == "bonus" else match["tossup"]
            return text, match["id"]

    pool = RAG_CHUNKS if category == "All Topics" else [c for c in RAG_CHUNKS if category.upper()[:4] in c["tossup"].upper()]
    if not pool:
        pool = RAG_CHUNKS
    unused = [c for c in pool if c["id"] not in USED_CHUNK_IDS]
    if not unused:
        USED_CHUNK_IDS.clear()
        unused = pool
    chosen = random.choice(unused)
    USED_CHUNK_IDS.add(chosen["id"])
    text = chosen["bonus"] if q_type.lower() == "bonus" else chosen["tossup"]
    return text, chosen["id"]

def create_question_payload(selected_category="All Topics", q_type="Toss-up", mode="hybrid", chunk_id=None):
    if mode == "mix":
        mode = random.choice(["exact", "hybrid"])
    recent_avoid = "; ".join(ASKED_HISTORY[-12:]) if ASKED_HISTORY else "None"

    if mode == "exact":
        raw_chunk, used_id = get_random_chunk(selected_category, q_type, chunk_id)
        if not raw_chunk:
            raise ValueError("No PDFs found in 'packets' folder! Upload DOE PDFs or switch mode to AI.")
        extract_prompt = (
            f"Extract the {q_type} question, its subject category (Biology, Chemistry, Physics, Mathematics, Earth and Space, or Energy), "
            f"and its official ANSWER word-for-word from this Science Bowl text. Keep exact wording and options. Format math with $...$.\n\nText:\n{raw_chunk}"
        )
        result = call_gemini_fast(extract_prompt, QuestionData, 0.1)
        if selected_category != "All Topics":
            result["category"] = selected_category
        result["q_type"] = q_type
        result["source"] = "Exact PDF"
        result["chunk_id"] = used_id
        ASKED_HISTORY.append(f"{result['question'][:50]} ({result['answer']})")
        return result

    actual_category = random.choice(list(SUBTOPICS.keys())) if selected_category == "All Topics" else selected_category
    subtopic = random.choice(SUBTOPICS.get(actual_category, ["General Science"]))
    q_format = random.choice(["Multiple Choice (with W, X, Y, Z options)", "Short Answer"])

    prompt = (
        f"You are an expert National Science Bowl writer. Generate 1 unique High School {q_type} question. "
        f"Category: {actual_category}. Subtopic: {subtopic}. Format: {q_format}. "
        f"CRITICAL RULE: If the question format is Multiple Choice or includes phrases like 'which of the following', you MUST explicitly provide options W, X, Y, and Z as part of the question text. "
        f"Format any math equations with KaTeX $...$ notation. Do NOT repeat: [{recent_avoid}]."
    )
    result = call_gemini_fast(prompt, QuestionData, 0.9)
    result["category"] = actual_category
    result["q_type"] = q_type
    result["source"] = "AI"
    result["chunk_id"] = None
    ASKED_HISTORY.append(f"{result['question'][:50]} ({result['answer']})")
    return result

def normalize_ans(text):
    text = re.sub(r'\(.*?\)', '', text.lower())
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = [w for w in text.split() if w not in ('a', 'an', 'the')]
    return ' '.join(words).strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        uploaded_files = request.files.getlist('file')
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({"error": "No selected files"}), 400

        os.makedirs("packets", exist_ok=True)
        saved_count = 0
        for file in uploaded_files:
            if file and file.filename.endswith('.pdf'):
                file.save(os.path.join("packets", file.filename))
                saved_count += 1

        if saved_count > 0:
            global RAG_CHUNKS
            RAG_CHUNKS = load_rag_chunks()
            return jsonify({"success": True, "message": f"Indexed {saved_count} PDF packet(s)! Total blocks: {len(RAG_CHUNKS)}"})
        return jsonify({"error": "No valid PDF files uploaded"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate_question():
    try:
        data = request.json or {}
        result = create_question_payload(
            data.get("category", "All Topics"),
            data.get("q_type", "Toss-up"),
            data.get("mode", "hybrid"),
            data.get("chunk_id", None)
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/grade", methods=["POST"])
def grade_answer():
    try:
        data = request.json or {}
        q = data.get('question', '')
        official = data.get('official_answer', '')
        user_ans = data.get('user_answer', '')

        norm_user = normalize_ans(user_ans)
        norm_off = normalize_ans(official)

        if norm_user and (norm_user == norm_off or (len(norm_user) == 1 and norm_off.startswith(norm_user))):
            return jsonify({"correct": True, "explanation": "Exact match."})

        prompt = (
            f"You are an official National Science Bowl judge. Question: {q} | "
            f"Official Answer: {official} | Student Answer: {user_ans}. "
            f"Determine if the student answer is scientifically equivalent or matches the correct option, and give a concise 1-sentence explanation."
        )
        return jsonify(call_gemini_fast(prompt, GradeData, 0.1))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- CONTEST ROUTES ---
@app.route("/contest/create", methods=["POST"])
def create_contest():
    try:
        data = request.json or {}
        team_name = data.get("team_name", "Team A").strip() or "Team A"
        settings = data.get("settings") or {
            "category": "All Topics",
            "practice_format": "Combined",
            "source_mode": "hybrid",
            "timing_mode": "timed"
        }

        contest_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        password = ''.join(random.choices(string.digits, k=4))

        CONTESTS[contest_id] = {
            "password": password,
            "creator": team_name,
            "settings": settings,
            "teams": {team_name: 0},
            "ready_teams": [],
            "question": None,
            "buzzed_team": None,
            "wrong_teams": [],
            "bonus_team": None,
            "bonus_category": None,
            "bonus_chunk_id": None,
            "status": "waiting",
            "last_result": f"Room created by {team_name}. Waiting for players to join and press Next Toss-up..."
        }
        return jsonify({"contest_id": contest_id, "password": password, "state": CONTESTS[contest_id]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/contest/join", methods=["POST"])
def join_contest():
    try:
        data = request.json or {}
        contest_id = data.get("contest_id", "").strip().upper()
        password = data.get("password", "").strip()
        team_name = data.get("team_name", "").strip()

        if not team_name:
            return jsonify({"error": "Please enter a Team Name."}), 400
        if contest_id not in CONTESTS:
            return jsonify({"error": "Contest ID not found."}), 404
        room = CONTESTS[contest_id]
        if room["password"] != password:
            return jsonify({"error": "Incorrect Contest Password."}), 403

        if team_name not in room["teams"]:
            room["teams"][team_name] = 0
            room["last_result"] = f"{team_name} joined the contest! Both teams must press Next Toss-up to begin."

        return jsonify({"contest_id": contest_id, "state": room})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/contest/settings", methods=["POST"])
def update_contest_settings():
    try:
        data = request.json or {}
        contest_id = data.get("contest_id", "").strip().upper()
        team_name = data.get("team_name", "").strip()
        if contest_id not in CONTESTS:
            return jsonify({"error": "Contest not found"}), 404
        room = CONTESTS[contest_id]

        if room["creator"] != team_name:
            return jsonify({"error": "Only the room creator can change settings."}), 403

        room["settings"] = data.get("settings", room["settings"])
        return jsonify(room)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/contest/state/<contest_id>", methods=["GET"])
def get_contest_state(contest_id):
    contest_id = contest_id.strip().upper()
    if contest_id not in CONTESTS:
        return jsonify({"error": "Contest not found"}), 404
    return jsonify(CONTESTS[contest_id])

@app.route("/contest/next", methods=["POST"])
def contest_next_question():
    try:
        data = request.json or {}
        contest_id = data.get("contest_id", "").strip().upper()
        team_name = data.get("team_name", "").strip()
        if contest_id not in CONTESTS:
            return jsonify({"error": "Contest not found"}), 404

        room = CONTESTS[contest_id]
        settings = room["settings"]
        practice_format = settings.get("practice_format", "Combined")
        mode = settings.get("source_mode", "hybrid")

        # Determine question type from creator's Practice Format setting
        requested_type = data.get("q_type", "Toss-up")
        if requested_type == "Bonus" and room.get("bonus_team"):
            q_type = "Bonus"
        elif practice_format == "Bonus":
            q_type = "Bonus"
        else:
            q_type = "Toss-up"

        # Require all teams in the room to press Next Toss-up before launching a new Toss-up
        if q_type == "Toss-up":
            if team_name and team_name not in room["ready_teams"]:
                room["ready_teams"].append(team_name)

            all_teams = list(room["teams"].keys())
            waiting_on = [t for t in all_teams if t not in room["ready_teams"]]

            if len(waiting_on) > 0:
                room["last_result"] = f"⏳ {team_name} is ready. Waiting for {', '.join(waiting_on)} to press Next Toss-up..."
                return jsonify(room)

        # All teams are ready (or launching an earned Bonus) -> Generate and launch!
        room["ready_teams"] = []
        if q_type == "Bonus" and room.get("bonus_category"):
            category = room["bonus_category"]
            chunk_id = room.get("bonus_chunk_id")
        else:
            category = settings.get("category", "All Topics")
            chunk_id = None

        q_data = create_question_payload(category, q_type, mode, chunk_id)
        room["question"] = q_data
        room["buzzed_team"] = room["bonus_team"] if (q_type == "Bonus" and room.get("bonus_team")) else None
        room["wrong_teams"] = []
        room["status"] = "buzzed" if (q_type == "Bonus" and room.get("bonus_team")) else "active"
        room["last_result"] = f"New {q_type} ({q_data['category']}) live! " + (
            f"Bonus for {room['bonus_team']}!" if (q_type == "Bonus" and room.get("bonus_team")) else "Buzz in now!"
        )
        return jsonify(room)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/contest/buzz", methods=["POST"])
def contest_buzz():
    try:
        data = request.json or {}
        contest_id = data.get("contest_id", "").strip().upper()
        team_name = data.get("team_name", "").strip()

        if contest_id not in CONTESTS:
            return jsonify({"error": "Contest not found"}), 404
        room = CONTESTS[contest_id]

        if room["status"] == "active" and room["buzzed_team"] is None and team_name not in room.get("wrong_teams", []):
            room["buzzed_team"] = team_name
            room["status"] = "buzzed"
            room["last_result"] = f"{team_name} buzzed in!"
        return jsonify(room)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/contest/answer", methods=["POST"])
def contest_answer():
    try:
        data = request.json or {}
        contest_id = data.get("contest_id", "").strip().upper()
        team_name = data.get("team_name", "").strip()
        user_answer = data.get("user_answer", "").strip()

        if contest_id not in CONTESTS:
            return jsonify({"error": "Contest not found"}), 404
        room = CONTESTS[contest_id]
        q = room["question"]

        if not q or room["buzzed_team"] != team_name:
            return jsonify({"error": "It is not your turn to answer!"}), 400

        norm_user = normalize_ans(user_answer)
        norm_off = normalize_ans(q['answer'])

        if norm_user and (norm_user == norm_off or (len(norm_user) == 1 and norm_off.startswith(norm_user))):
            is_correct = True
            explanation = "Exact match."
        else:
            prompt = (
                f"You are an official National Science Bowl judge. Question: {q['question']} | "
                f"Official Answer: {q['answer']} | Student Answer: {user_answer}. "
                f"Determine if the student answer is scientifically equivalent or matches the correct option, and give a concise 1-sentence explanation."
            )
            grade = call_gemini_fast(prompt, GradeData, 0.1)
            is_correct = grade["correct"]
            explanation = grade["explanation"]

        is_bonus = (q["q_type"].lower() == "bonus")
        practice_format = room["settings"].get("practice_format", "Combined")
        total_teams = len(room["teams"])

        if is_correct:
            points = 10 if is_bonus else 4
            room["teams"][team_name] = room["teams"].get(team_name, 0) + points
            room["status"] = "answered"
            room["ready_teams"] = []
            if is_bonus or practice_format == "Toss-up":
                room["bonus_team"] = None
                room["bonus_category"] = None
                room["bonus_chunk_id"] = None
            else:
                room["bonus_team"] = team_name
                room["bonus_category"] = q["category"]
                room["bonus_chunk_id"] = q.get("chunk_id")
            room["last_result"] = f"✅ {team_name} is CORRECT (+{points} pts)! Answer: {q['answer']}. {explanation}"
        else:
            room["wrong_teams"].append(team_name)
            if is_bonus or len(room["wrong_teams"]) >= total_teams:
                room["status"] = "answered"
                room["ready_teams"] = []
                room["bonus_team"] = None
                room["bonus_category"] = None
                room["bonus_chunk_id"] = None
                room["last_result"] = f"❌ Question Dead! Official Answer: {q['answer']}. {explanation}"
            else:
                room["buzzed_team"] = None
                room["status"] = "active"
                room["last_result"] = f"❌ {team_name} is INCORRECT! Other teams can buzz in!"

        return jsonify(room)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)