# Adobe India Hackathon 2025 — Challenge 1B  
**Persona-Driven Document Intelligence – Methodology Explanation**

## 📌 Objective

The goal is to extract highly relevant content from multiple PDFs using a persona-task input and return two things:
- Top relevant **section titles** with document/page metadata.
- Detailed **refined analysis** (e.g., recipe breakdowns or instruction summaries).

The solution must:
- Run **on CPU** only
- Use a **model < 1GB**
- **Complete in ≤60s** for 3–5 PDFs
- **Not access the internet** during execution

---

## 🐳 Dockerization Strategy

### 🧪 Build
```bash
docker build -t adobe-hackathon-pipeline .
```

### 🚀 Run
```bash
docker run --rm -v "$(pwd):/app" -v "$(pwd)/models:/root/.cache/huggingface/transformers" adobe-hackathon-1b   
```

---

## 🧠 Approach Overview

### Step 1: 🧾 Input Interpretation
Each collection provides:
- A `challenge1b_input.json` file containing:
  - `"persona"` (e.g., *Food Contractor*)
  - `"job_to_be_done"` (e.g., *Prepare a vegetarian dinner menu*)
- A set of PDFs containing the source material

We combine the two into a **semantic query**:
```python
query = f"As a {persona}, I want to {task}."
```

This serves as the anchor for embedding-based similarity comparison.

---

### Step 2: 📄 PDF Text Extraction

Using `PyMuPDF (fitz)`, we extract raw text from **each page** (not entire PDF), preserving page numbers. This enables fine-grained analysis and source referencing:

```python
for page in doc:
    text = page.get_text("text")
    pages.append({"page": i + 1, "text": text})
```

Each page's text is stored per document, enabling section-level analysis.

---

### Step 3: 🔍 Semantic Embedding and Similarity Ranking

We use the compact `all-MiniLM-L6-v2` model from `sentence-transformers`:
- Size: ~90MB (well under 1GB)
- Performance: Strong semantic understanding, fast CPU execution

We encode:
- The combined persona-task **query**
- Each PDF page text
Then compute cosine similarity:

```python
score = util.pytorch_cos_sim(query_embedding, section_embedding)
```

Pages are ranked by similarity to the query. **Top 10 sections** are retained for further analysis.

---

### Step 4: 🧪 Section Title & Subsection Refinement

We extract:
- **Section Title**: First 1–2 lines of the top section
- **Subsection Analysis**:
  - Cleaned and converted to readable paragraph
  - If recipe-like, ingredients/instructions are split heuristically:
    ```python
    if "Ingredients" in text:
        summary = "This recipe includes ... To prepare, ..."
    ```

This ensures consistency across domains: travel, cooking, software guides, etc.

---




---

## 🧠 Design Strengths

- **Generalizable**: Works for travel guides, recipes, how-tos, etc.
- **Efficient**: Embedding + TFIDF reranking gives focused content
- **Readable**: Paragraph-style summaries, clean JSON
- **Modular**: Easily extendable for larger datasets or streaming input

---

## 📂 Directory Structure

```
Challenge_1b/
├── Collection 1/
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json (generated)
├── main.py
├── Dockerfile
└── approach_explanation.md
```

---

## ✅ Summary

The pipeline is light, robust, semantically intelligent, and production-ready. It satisfies every technical constraint and produces high-quality, interpretable output across document domains — from recipe curation to software tutorials.
