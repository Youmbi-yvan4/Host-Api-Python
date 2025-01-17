import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from docx import Document
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Text extraction functions
def extract_text_from_pdf(file_path):
    text = ""
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

def extract_text_from_txt(file_path):
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
    return text

def get_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lstrip('.').lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file extension: {ext}")
        return ""

# LCS similarity calculation
def lcs(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]

def calculate_lcs_similarity(text1, text2):
    words1, words2 = text1.split(), text2.split()
    if not words1 or not words2:
        return 0
    lcs_length = lcs(words1, words2)
    max_length = max(len(words1), len(words2))
    return (lcs_length / max_length) * 100

# Cosine similarity calculation
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0] * 100

# API Endpoints
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    file_paths = []
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_paths.append(file_path)
    return jsonify({"file_paths": file_paths}), 200

@app.route('/inspiration', methods=['POST'])
def inspiration():
    data = request.json
    file_paths = data.get('file_paths', [])
    if len(file_paths) != 2:
        return jsonify({"error": "Two file paths are required for inspiration calculation"}), 400
    text1 = get_text_from_file(file_paths[0])
    text2 = get_text_from_file(file_paths[1])
    if not text1 or not text2:
        return jsonify({"error": "Unable to extract text from one or both files"}), 400
    lcs_similarity = calculate_lcs_similarity(text1, text2)
    cosine_similarity = calculate_cosine_similarity(text1, text2)
    max_similarity = max(lcs_similarity, cosine_similarity)
    return jsonify({
        "lcs_similarity": lcs_similarity,
        "cosine_similarity": cosine_similarity,
        "max_similarity": max_similarity
    }), 200

@app.route('/similarities', methods=['POST'])
def similarities():
    data = request.json
    file_paths = data.get('file_paths', [])
    if len(file_paths) < 2:
        return jsonify({"error": "At least two file paths are required"}), 400
    texts = [get_text_from_file(path) for path in file_paths]
    if any(not text for text in texts):
        return jsonify({"error": "Unable to extract text from one or more files"}), 400
    results = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            lcs_sim = calculate_lcs_similarity(texts[i], texts[j])
            cosine_sim = calculate_cosine_similarity(texts[i], texts[j])
            max_sim = max(lcs_sim, cosine_sim)
            results.append({
                "files": [file_paths[i], file_paths[j]],
                "lcs_similarity": lcs_sim,
                "cosine_similarity": cosine_sim,
                "max_similarity": max_sim
            })
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)
