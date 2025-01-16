from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import string
from docx import Document
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
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
        return jsonify({"error": str(e)}), 500

@app.route('/inspiration', methods=['POST'])
def get_inspiration():
    try:
        data = request.json
        text1 = data.get('text1')
        text2 = data.get('text2')
        file_paths = data.get('file_paths', [])
        
        # Extract texts from files if file paths are provided
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
    except Exception as e:
        app.logger.error(f"Exception in inspiration API: {e}")
        return jsonify({"error": str(e)}), 500


def detect_inspirations(text1, text2):
    """
    Detects the inspiration percentage for each paragraph in `text2` based on its similarity to `text1`.
    """
    inspirations = []
    paragraphs1 = text1.split('\n')
    paragraphs2 = text2.split('\n')

    for para2 in paragraphs2:
        best_match_similarity = 0
        for para1 in paragraphs1:
            lcs_sim = calculate_similarity(para1, para2)
            cosine_sim = cosine_similarity_percentage(para1, para2)
            max_similarity = max(lcs_sim, cosine_sim)

            if max_similarity > 0:
                best_match_similarity = max(best_match_similarity, max_similarity)
        inspirations.append((para2, best_match_similarity))
    return inspirations


def calculate_total_inspiration(inspirations):
    """
    Calculate the overall inspiration percentage from the detected inspirations.
    """
    total_similarity_score = sum(similarity for _, similarity in inspirations)
    if len(inspirations) == 0:
        return 0
    total_inspiration_percentage = total_similarity_score / len(inspirations)
    return total_inspiration_percentage


def measure_inspiration(text1, text2):
    """
    Measures the overall inspiration of `text2` based on its similarity to `text1`.
    """
    inspirations = detect_inspirations(text1, text2)
    total_inspiration_percentage = calculate_total_inspiration(inspirations)
    return total_inspiration_percentage

def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        return extract_text_pdfplumber(file_path)
    elif file_extension == ".docx":
        return extract_text_docx(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        return extract_text_with_pandas(file_path)
    elif file_extension == ".txt":
        return extract_text_txt(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return None


def extract_text_pdfplumber(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages])
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None


def extract_text_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return None


def extract_text_with_pandas(file_path):
    try:
        data = pd.read_excel(file_path)
        return data.to_string(index=False)
    except Exception as e:
        print(f"Error extracting Excel data: {e}")
        return None


def extract_text_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting text from TXT file: {e}")
        return None


def preprocess_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))


def calculate_similarity(text1, text2):
    words1, words2 = text1.split(), text2.split()
    if not words1 or not words2:
        return 0

    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if X[i] == Y[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    lcs_length = lcs(words1, words2)
    max_length = max(len(words1), len(words2))
    return (lcs_length / max_length) * 100 if max_length > 0 else 0


def cosine_similarity_percentage(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0] * 100


def calculate_and_display_similarities(documents):
    similarities = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            lcs_sim = calculate_similarity(documents[i], documents[j])
            cosine_sim = cosine_similarity_percentage(documents[i], documents[j])
            best_similarity = max(lcs_sim, cosine_sim)
            similarities.append({"documents": [f"Doc{i+1}", f"Doc{j+1}"], "similarity": best_similarity})
    return similarities


if __name__ == '__main__':
    app.run(debug=True)
