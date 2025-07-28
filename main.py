import os
import json
import fitz  # PyMuPDF
import datetime
import re
import heapq
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

model = SentenceTransformer("./models/all-MiniLM-L6-v2")


def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
    doc.close()
    return pages

def split_paragraphs(pages):
    chunks = []
    for i, page in enumerate(pages):
        paras = re.split(r"\n{2,}|(?<=\.)\s{2,}", page)
        for para in paras:
            clean = para.strip()
            if len(clean) > 50:
                chunks.append((i + 1, clean))
    return chunks

def get_section_title(text):
    candidates = re.split(r"\n{1,}", text)
    for line in candidates:
        line = line.strip()
        if 5 < len(line) < 100 and line[0].isupper():
            return re.sub(r"[:\-–\n]+$", "", line)
    return text[:80].strip()

def rerank_with_tfidf(chunks, top_k):
    texts = [c[1] for c in chunks]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    scores = sim_matrix.sum(axis=1)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def get_top_chunks(pages, persona, task, top_k=5):
    query = f"As a {persona}, I want to {task}."
    query_emb = model.encode(query, convert_to_tensor=True)
    chunks = split_paragraphs(pages)
    texts = [c[1] for c in chunks]
    emb = model.encode(texts, convert_to_tensor=True)
    sim_scores = util.pytorch_cos_sim(query_emb, emb)[0].cpu().numpy()
    top_idx = heapq.nlargest(top_k * 3, range(len(sim_scores)), sim_scores.take)
    top_chunks = [chunks[i] for i in top_idx]
    reranked = rerank_with_tfidf(top_chunks, top_k)
    return reranked

def convert_to_paragraph(text):
    # Normalize bullets and whitespace
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[•▪◦‣•●∙·]", "-", text)
    text = re.sub(r"\bo\b", "-", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", text)

    # Try to extract ingredients and instructions
    match = re.search(r"(Ingredients.*?)(Instructions[:\-]?)?(.*)", text, re.IGNORECASE)
    if match:
        ing = re.sub(r"[-•▪◦‣•●∙·]", "", match.group(1)).strip()
        inst = re.sub(r"[-•▪◦‣•●∙·]", "", match.group(3)).strip()
        if inst:
            return f"This recipe includes {ing}. To prepare, {inst[0].lower() + inst[1:]}"
        else:
            return f"This recipe includes {ing}."
    
    return text



def process_document(doc, persona, task, pdf_dir):
    filename = doc["filename"]
    pdf_path = os.path.join(pdf_dir, filename)
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return [], []
    pages = extract_text_by_page(pdf_path)
    top_chunks = get_top_chunks(pages, persona, task, top_k=5)

    extracted = []
    analysis = []

    for i, (page, para) in enumerate(top_chunks):
        extracted.append({
            "document": filename,
            "section_title": get_section_title(para),
            "importance_rank": i + 1,
            "page_number": page
        })
        # analysis.append({
        #     "document": filename,
        #     "refined_text": para,
        #     "page_number": page
        # })
        # Clean and condense the paragraph text
        cleaned = re.sub(r"\n+", " ", para)                  # Remove line breaks
        cleaned = re.sub(r"•", "-", cleaned)                 # Replace bullet points
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()    # Remove extra spaces
        cleaned = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", cleaned)  # Add space after periods if missing

        # Try to keep only useful sentences (like ingredient lists or instructions)
        if len(cleaned.split()) > 150:
            cleaned = " ".join(cleaned.split()[:150]) + "..."

        refined = convert_to_paragraph(para)
        analysis.append({
            "document": filename,
            "refined_text": refined,
            "page_number": page
        })



    return extracted, analysis

def run_pipeline(input_path, output_path, pdf_folder):
    if not os.path.exists(input_path):
        print(f"❌ Input file missing: {input_path}")
        return
    if not os.path.exists(pdf_folder):
        print(f"❌ PDF folder missing: {pdf_folder}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    persona = input_json["persona"]["role"]
    task = input_json["job_to_be_done"]["task"]
    documents = input_json["documents"]

    all_sections = []
    all_analysis = []

    for doc in tqdm(documents, desc=f"Processing {os.path.basename(input_path)}"):
        sections, subs = process_document(doc, persona, task, pdf_folder)
        all_sections.extend(sections)
        all_analysis.extend(subs)

    # De-duplicate & select top-5 globally across all documents
    all_sections = sorted(all_sections, key=lambda x: x["importance_rank"])
    for i, sec in enumerate(all_sections[:5]):
        sec["importance_rank"] = i + 1

    all_analysis = sorted(all_analysis, key=lambda x: x["page_number"])

    output = {
        "metadata": {
            "input_documents": [d["filename"] for d in documents],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": str(datetime.datetime.now())
        },
        "extracted_sections": all_sections[:5],
        "subsection_analysis": all_analysis[:5]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Written: {output_path}")

if __name__ == "__main__":
    base = "."  # since main.py is in Challenge_1b
    collections = ["Collection 1", "Collection 2", "Collection 3"]

    for coll in collections:
        input_path = os.path.join(base, coll, "challenge1b_input.json")
        output_path = os.path.join(base, coll, "challenge1b_output.json")
        pdf_path = os.path.join(base, coll, "PDFs")
        run_pipeline(input_path, output_path, pdf_path)
