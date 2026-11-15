import os
import json
import random
import glob
import traceback
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader

# This checks if a cloud environment variable exists; if not, it uses your local fallback string
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "YourLocalApiKeyHere")

app = Flask(__name__, template_folder=".")

client = genai.Client(
    http_options=types.HttpOptions(
        timeout=10000,
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)

def get_available_models():
    models = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name and ("flash" in name or "pro" in name) and "tts" not in name and "image" not in name:
                models.append(name)
    except Exception as e:
        print("Could not list models:", e)
    models.sort(key=lambda x: (0 if "lite" in x else (1 if "flash" in x else 2), x))
    return models if models else ["gemini-2.5-flash", "gemini-2.0-flash"]

MODELS_TO_TRY = get_available_models()

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
    print(f"=== RAG LOADED & SHUFFLED: {len(chunks)} paired question blocks from {len(pdf_files)} PDFs ===")
    return chunks

RAG_CHUNKS = load_rag_chunks()
USED_CHUNK_IDS = set()

def get_random_chunk(category, q_type):
    if not RAG_CHUNKS:
        return None
    pool = RAG_CHUNKS if category == "All Topics" else [c for c in RAG_CHUNKS if category.upper()[:4] in c["tossup"].upper()]
    if not pool:
        pool = RAG_CHUNKS
    
    unused = [c for c in pool if c["id"] not in USED_CHUNK_IDS]
    if not unused:
        USED_CHUNK_IDS.clear()
        unused = pool
        
    chosen = random.choice(unused)
    USED_CHUNK_IDS.add(chosen["id"])
    return chosen["bonus"] if q_type.lower() == "bonus" else chosen["tossup"]

ASKED_HISTORY = []

SUBTOPICS = {
    "Biology": ["Cell Biology", "Genetics & Evolution", "Human Anatomy & Physiology", "Botany & Plant Science", "Ecology", "Microbiology", "Biochemistry"],
    "Chemistry": ["Stoichiometry", "Thermochemistry", "Organic Chemistry", "Acids & Bases", "Electrochemistry", "Periodic Trends", "Chemical Kinetics", "Nuclear Chemistry"],
    "Physics": ["Kinematics & Mechanics", "Electricity & Magnetism", "Optics & Waves", "Thermodynamics", "Quantum & Modern Physics", "Fluid Mechanics", "Rotational Motion"],
    "Mathematics": ["Algebra & Polynomials", "Geometry & Trigonometry", "Combinatorics & Probability", "Number Theory", "Precalculus & Sequences", "Introductory Calculus"],
    "Earth and Space": ["Astronomy & Stars", "Planetary Science", "Plate Tectonics & Geology", "Meteorology & Climate", "Oceanography", "Mineralogy & Rocks"],
    "Energy": ["Renewable Energy (Solar/Wind)", "Nuclear Fission & Fusion", "Batteries & Energy Storage", "Fossil Fuels & Thermodynamics", "Grid & Power Transmission", "DOE National Labs Research"]
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

def call_gemini_with_fallback(prompt, schema, temp):
    last_error = None
    for model_name in MODELS_TO_TRY[:4]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temp
                )
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e
            continue
    raise last_error

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
                filepath = os.path.join("packets", file.filename)
                file.save(filepath)
                saved_count += 1
                
        if saved_count > 0:
            global RAG_CHUNKS
            RAG_CHUNKS = load_rag_chunks()
            return jsonify({"success": True, "message": f"Successfully uploaded and indexed {saved_count} PDF packet(s)!"})
        
        return jsonify({"error": "No valid PDF files were uploaded"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate_question():
    try:
        data = request.json
        selected_category = data.get("category", "All Topics")
        q_type = data.get("q_type", "Toss-up")
        mode = data.get("mode", "hybrid")
        
        if mode == "mix":
            mode = random.choice(["exact", "hybrid"])
            
        recent_avoid = "; ".join(ASKED_HISTORY[-15:]) if ASKED_HISTORY else "None"

        if mode == "exact":
            raw_chunk = get_random_chunk(selected_category, q_type)
            if not raw_chunk:
                return jsonify({"error": "No PDFs found in your 'packets' folder! Upload official DOE PDFs above or switch mode to 'AI'."}), 400
            
            extract_prompt = (
                f"Extract the {q_type} question, its subject category (Biology, Chemistry, Physics, Mathematics, Earth and Space, or Energy), and its official ANSWER word-for-word from this National Science Bowl packet text. "
                f"Do not invent a new question—keep the exact wording and options from the text.\n\nText:\n{raw_chunk}"
            )
            result = call_gemini_with_fallback(extract_prompt, QuestionData, 0.1)
            if selected_category != "All Topics":
                result["category"] = selected_category
            result["q_type"] = q_type
            result["source"] = "Exact PDF"
            ASKED_HISTORY.append(f"{result['question'][:60]} ({result['answer']})")
            return jsonify(result)

        actual_category = random.choice(list(SUBTOPICS.keys())) if selected_category == "All Topics" else selected_category
        subtopic = random.choice(SUBTOPICS.get(actual_category, ["General Science"]))
        q_format = random.choice(["Multiple Choice (with W, X, Y, Z options)", "Short Answer"])
        
        prompt = (
            f"You are an expert National Science Bowl question writer. Generate 1 unique High School {q_type} question. "
            f"Category: {actual_category}. Specific Subtopic: {subtopic}. Format: {q_format}. "
            f"Do NOT repeat any of these previously asked questions: [{recent_avoid}]."
        )
        
        result = call_gemini_with_fallback(prompt, QuestionData, 1.0)
        result["category"] = actual_category
        result["q_type"] = q_type
        result["source"] = "AI"
        ASKED_HISTORY.append(f"{result['question'][:60]} ({result['answer']})")
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/grade", methods=["POST"])
def grade_answer():
    try:
        data = request.json
        prompt = f"You are an official National Science Bowl judge. Question: {data['question']} | Official Answer: {data['official_answer']} | Student Answer: {data['user_answer']}. Determine if the student answer is scientifically equivalent or matches the correct option, and give a 1-sentence explanation."
        return jsonify(call_gemini_with_fallback(prompt, GradeData, 0.1))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)