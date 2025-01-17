from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import logging
import traceback
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            app.logger.error("No files part in the request")
            return jsonify({"error": "No files part in the request"}), 400
        
        files = request.files.getlist('files')
        if not files:
            app.logger.error("No files uploaded")
            return jsonify({"error": "No files uploaded"}), 400
        
        file_urls = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_urls.append(file_path)
        
        return jsonify({"file_urls": file_urls}), 200
    except Exception as e:
        app.logger.error(f"Exception occurred during file upload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/similarities', methods=['POST'])
def get_similarities():
    try:
        data = request.json
        file_paths = data.get('file_paths', [])
        
        if not file_paths or len(file_paths) < 2:
            app.logger.error("Not enough file paths to compare")
            return jsonify({"error": "Not enough file paths to compare."}), 400
        
        # Extract texts from files
        documents = []
        for file_path in file_paths:
            text = extract_text_from_file(file_path)
            if text:
                documents.append(text)
        
        if len(documents) < 2:
            app.logger.error("Not enough extracted texts to compare")
            return jsonify({"error": "Not enough extracted texts to compare."}), 400

        similarity_results = calculate_and_display_similarities(documents)
        return jsonify(similarity_results), 200
    except Exception as e:
        app.logger.error(f"Exception occurred: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/inspiration', methods=['POST'])
def get_inspiration():
    data = request.json
    text1 = data.get('text1')
    text2 = data.get('text2')
    file_paths = data.get('file_paths', [])
    
    # Extract texts from files if provided
    for file_path in file_paths:
        extracted_text = extract_text_from_file(file_path)
        if extracted_text:
            if not text1:
                text1 = extracted_text
            elif not text2:
                text2 = extracted_text
            else:
                # Combine extracted texts if both text1 and text2 are already provided
                text2 += "\n" + extracted_text
    
    if not text1 or not text2:
        return jsonify({"error": "Both text1 and text2 are required either directly or via file paths."}), 400
    
    inspiration_percentage = measure_inspiration(text1, text2)
    return jsonify({"inspiration_percentage": inspiration_percentage}), 200

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

def extract_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lstrip('.').lower()

    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, 'r') as file:
            return file.read()

def preprocess_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def segment_text(text):
    return text.split()

def lcs(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def calculate_lcs_similarity(text1, text2):
    words1 = segment_text(preprocess_text(text1))
    words2 = segment_text(preprocess_text(text2))
    lcs_length = lcs(words1, words2)
    return (lcs_length / len(words1)) * 100 if words1 else 0

def cosine_similarity_percentage(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0] * 100

def measure_inspiration(text1, text2):
    lcs_sim = calculate_lcs_similarity(text1, text2)
    cosine_sim = cosine_similarity_percentage(text1, preprocess_text(text2))

    max_similarity = max(lcs_sim, cosine_sim)
    return max_similarity

if __name__ == '__main__':
    app.run(debug=True)
