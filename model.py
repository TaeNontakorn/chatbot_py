from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
import os
import shutil
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ โหลดค่าจากไฟล์ .env
load_dotenv()

# 🔑 ตั้งค่า Token และโมเดล
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-Small-24B-Instruct-2501"
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "The_Python_Book.pdf"

# ✅ ตั้งค่า Flask และเปิด CORS
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ✅ โหลดเอกสารจาก PDF
def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ PDF: {file_path}")
    
    loader = PyMuPDFLoader(file_path)
    return loader.load()

documents = load_pdf(DATA_PATH)

# ✅ แบ่งเอกสารเป็น chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
text_chunks = splitter.split_documents(documents)

# ✅ สร้าง embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# ✅ ลบ FAISS Index เก่าและสร้างใหม่เสมอ
if os.path.exists(DB_FAISS_PATH):
    shutil.rmtree(DB_FAISS_PATH, ignore_errors=True)
os.makedirs(DB_FAISS_PATH, exist_ok=True)

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

# ✅ โหลดโมเดล LLM ผ่าน InferenceClient
client = InferenceClient(api_key=HF_TOKEN)

# ✅ ฟังก์ชันสำหรับค้นคืนข้อมูลและตอบคำถามจากไฟล์เท่านั้น
def generate_response(query):
    context_docs = db.similarity_search_with_score(query, k=3)  
    if not context_docs or context_docs[0][1] < 0.5:  
        return "I don't know. The answer is not found in the provided document."

    combined_context = "\n\n".join([doc.page_content for doc, score in context_docs])

    refined_prompt = f"""
    You are an AI assistant that strictly answers questions based on the given document.
    Do not make up information. If the answer is not found in the document, please say so.
    If the answer is found in the document, provide the answer with the relevant context.
    If the answer is not found in the document,try to guess the answer based on the context.
    If you are not sure, please say so.

    Context:
    {combined_context}

    Question:
    {query}

    Answer:
    """

    result = client.text_generation(prompt=refined_prompt, max_new_tokens=100, temperature=0.1)
    return result

# ✅ เสิร์ฟไฟล์ HTML (Frontend)
@app.route("/")
def index():
    return render_template("index.html")  # ✅ ตรวจสอบว่ามีไฟล์นี้ใน /templates

# ✅ API สำหรับรับคำถามจาก Frontend
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        query = data.get("question", "").strip()

        if not query:
            return jsonify({"error": "No question provided"}), 400

        response = generate_response(query)
        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
