import os
import re
import json
import uuid
import time
import base64
import traceback
import redis

from dotenv import load_dotenv
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from ollama import Client

from bank_upload_module.bank_profiles import resolve_bank_profile
from bank_upload_module.db import save_to_database

load_dotenv()
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
TEMP_DIR = os.environ.get("TEMP_DIR", "temp")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "bank_ocr_queue")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:3b")

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL","all-MiniLM-L6-v2")
POPPLER_PATH = os.environ.get("POPPLER_PATH") or None
PDF_DPI = int(os.environ.get("PDF_DPI", "100"))

CLUSTER_DISTANCE_THRESHOLD = float(os.environ.get("CLUSTER_DISTANCE_THRESHOLD", "0.40"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

redis_client = redis.Redis(host=REDIS_HOST,port=REDIS_PORT,decode_responses=True)
client = Client(host=OLLAMA_HOST)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

TRANSACTION_SCHEMA_KEYS = [
    "transaction_date","value_date","narration","withdrawal","deposit","balance","merchant_name","group_key"]
def pdf_to_images(pdf_path):

    images = convert_from_path(pdf_path,dpi=PDF_DPI,poppler_path=POPPLER_PATH)
    image_paths = []

    for i, image in enumerate(images):
        image_path = os.path.join(TEMP_DIR,f"{uuid.uuid4()}_{i}.png")
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
        
def clean_json_response(content):
    if not content:
        raise Exception("Empty response from model")
    content = content.replace("```json", "")
    content = content.replace("```", "")
    return content.strip()

def normalize_transaction(txn):
    normalized = {}
    for key in TRANSACTION_SCHEMA_KEYS:
        normalized[key] = str(txn.get(key, "")).strip()
    return normalized

def build_extraction_prompt(bank_name):
    profile = resolve_bank_profile(bank_name)
    if profile:
        bank_block = f"""
This statement belongs to {bank_name} Bank.

Table Headers:
{", ".join(profile.get("table_headers", []))}

Date Format:
{profile.get("date_format")}

Layout Notes:
{profile.get("layout_notes")}
"""
    else:
        bank_block = """
Unknown bank.
Infer layout automatically.
"""
    return f"""
You are an expert OCR engine.
{bank_block}
Extract ALL transactions.
Return ONLY VALID JSON.
Do NOT return markdown.
Return exactly this schema.
{{
    "statement": {{
        "bank_ledger":""
    }},
    "transactions":[
        {{"transaction_date":"","value_date":"","narration":"","withdrawal":"","deposit":"","balance":"","merchant_name":"","group_key":""}}
    ]
}}
"""
def qwen_extract_transactions(image_path, bank_name):
    prompt = build_extraction_prompt(bank_name)
    response = client.chat(
        model=VISION_MODEL,
        messages=[
            {"role": "user","content": prompt,"images": [image_path]}
        ],

        options={"temperature": 0,"top_p": 0.1,"num_ctx": OLLAMA_NUM_CTX,"num_predict": 4096}
    )
    raw = response["message"]["content"].strip()
    print("=" * 60)
    print("RAW MODEL RESPONSE")
    print(raw)
    print("=" * 60)

    cleaned = clean_json_response(raw)
    data = json.loads(cleaned)
    statement = {}
    transactions = []

    if isinstance(data, dict):
        statement = data.get("statement", {})
        transactions = data.get("transactions", [])
    elif isinstance(data, list):
        transactions = data

    normalized = []
    for txn in transactions:
        normalized.append(normalize_transaction(txn))

    return {"statement": statement,"transactions": normalized}
    
def read_document(file_path, bank_name):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        image_paths = pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    final_result = {"statement": {},"transactions": []}

    total_pages = len(image_paths)
    print(f"TOTAL PAGES : {total_pages}")
    for page_no, image_path in enumerate(image_paths, start=1):
        print(f"Processing Page {page_no}")

        try:
            page_result = qwen_extract_transactions(image_path, bank_name)
            if page_no == 1:
                final_result["statement"] = page_result.get("statement", {})

            final_result["transactions"].extend(page_result.get("transactions", []))

        except Exception as e:
            print("PAGE EXTRACTION ERROR")
            print(e)
            traceback.print_exc()
    return final_result
# ----------------------------------------------------------------------
# NARRATION CLEANING
# ----------------------------------------------------------------------
def clean_narration(text):
    if text is None:
        return ""
    text = text.upper()
    text = re.sub(r"@[A-Z0-9._]+", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)
    text = re.sub(r"[^A-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def create_embeddings(transactions):
    texts = [
        clean_narration(txn.get("narration", ""))
        for txn in transactions
    ]
    if not texts:
        return []
    return embedding_model.encode(texts,convert_to_numpy=True)

def cluster_embeddings(embeddings):
    if len(embeddings) <= 1:
        return [0] * len(embeddings)

    clustering = AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=CLUSTER_DISTANCE_THRESHOLD)
    return clustering.fit_predict(embeddings)

def group_transactions(transactions):
    print("Creating embeddings...")
    embeddings = create_embeddings(transactions)
    print("Embeddings Created.")
    if len(embeddings) == 0:
        return {}

    print("Running clustering...")
    labels = cluster_embeddings(embeddings)
    print("Grouping Completed.")
    groups = {}

    for txn, label in zip(transactions, labels):
        group = f"group_{label}"
        txn["group_key"] = group
        groups.setdefault(group, []).append(txn)
    return groups

def save_output(job_id, result):
    output_path = os.path.join(OUTPUT_DIR,f"{job_id}.json")

    print(f"Saving JSON -> {output_path}")
    with open(output_path,"w",encoding="utf-8") as f:

        json.dump(result,f,indent=4,ensure_ascii=False)
    print("JSON Saved Successfully.")

def cleanup_temp():
    if not os.path.exists(TEMP_DIR):
        return
    for file in os.listdir(TEMP_DIR):
        path = os.path.join(TEMP_DIR, file)
        try:
            os.remove(path)
        except Exception:
            pass

def process_document(job):
    job_id = job["job_id"]
    bank_name = job.get("bank_name", "")
    file_name = job.get("file_name", "")
    file_path = job["file_path"]
    redis_client.set(f"status:{job_id}","processing")

    print("=" * 70)
    print(f"JOB ID : {job_id}")
    print(f"BANK   : {bank_name}")
    print(f"FILE   : {file_path}")
    print("=" * 70)
    result = read_document(file_path,bank_name)
    statement = result.get("statement",{})
    bank_ledger = statement.get("bank_ledger","")
    transactions = result.get("transactions",[])
    if len(transactions) == 0:
        raise Exception("The uploaded PDF is empty. Please upload a valid bank statement.")

    print(f"Transactions : {len(transactions)}")

    groups = {}
    if transactions:
        groups = group_transactions(transactions)

    matched_profile = resolve_bank_profile(bank_name)
    result["document_id"] = job_id
    result["bank_name"] = bank_name
    result["bank_profile_matched"] = bool(matched_profile)
    result["total_transactions"] = len(transactions)
    result["groups"] = groups
    result["status"] = "Completed"

    print("=" * 70)
    print("Saving JSON")
    print("=" * 70)
    save_output(job_id,result)

    print("=" * 70)
    print("Saving PostgreSQL")
    print("=" * 70)
    save_to_database(job_id=job_id,bank_name=bank_name,file_name=file_name,bank_ledger=bank_ledger,result=result)

    redis_client.set(f"status:{job_id}","completed")
    print("=" * 70)
    print("JOB COMPLETED")
    print("=" * 70)

    cleanup_temp()
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(e)

def worker():

    print("=" * 70)
    print("BANK OCR WORKER STARTED")
    print(f"POPPLER_PATH : {POPPLER_PATH or '(using system PATH)'}")
    print(f"OLLAMA HOST  : {OLLAMA_HOST}")
    print(f"VISION MODEL : {VISION_MODEL}")
    print("=" * 70)
    while True:
        try:
            job = redis_client.lpop(QUEUE_NAME)
            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            job = json.loads(job)
            job_id = job["job_id"]
            print()
            print("=" * 70)
            print(f"NEW JOB RECEIVED : {job_id}")
            print("=" * 70)

            try:
                process_document(job)
                print()
                print("=" * 70)
                print(f"{job_id} COMPLETED SUCCESSFULLY")
                print("=" * 70)

            except Exception as e:
                print()
                print("=" * 70)
                print("PROCESSING FAILED")
                print("=" * 70)
                print(e)

                traceback.print_exc()
                redis_client.set(f"status:{job_id}","failed")

                error_file = os.path.join(OUTPUT_DIR,f"{job_id}_error.txt")

                with open(error_file,"w",encoding="utf-8") as f:
                    f.write(str(e))
                    f.write("\n\n")
                    f.write(traceback.format_exc())

                cleanup_temp()
        except KeyboardInterrupt:
            print()
            print("=" * 70)
            print("WORKER STOPPED")
            print("=" * 70)
            break

        except Exception as e:
            print()
            print("=" * 70)
            print("WORKER ERROR")
            print("=" * 70)
            print(e)

            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    worker()