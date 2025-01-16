from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from docx import Document
import pdfplumber
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
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
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def extract_text_docx(file_path):
    try:
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return None

def extract_text_with_pandas(file_path):
    try:
        data = pd.read_excel(file_path)  # Reads the first sheet by default
        return data.to_string(index=False)  # Convert the DataFrame to string for printing
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

def calculate_and_display_similarities(documents):
    def get_range(percentage):
        if percentage >= 90:
            return "90-100%"
        elif percentage >= 80:
            return "80-90%"
        elif percentage >= 70:
            return "70-80%"
        elif percentage >= 60:
            return "60-70%"
        elif percentage >= 50:
            return "50-60%"
        elif percentage >= 40:
            return "40-50%"
        elif percentage >= 30:
            return "30-40%"
        elif percentage >= 20:
            return "20-30%"
        elif percentage >= 10:
            return "10-20%"
        else:
            return "0-10%"

    similarities = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            similarity = structural_similarity(documents[i], documents[j])
            range_label = get_range(similarity)
            similarities.append([
                f"Document {i + 1} and Document {j + 1}",
                similarity,
                range_label
            ])
    return similarities

def preprocess_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def segment_text(text):
    return text.split('\n')

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[None]*(n+1) for i in range(m+1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    return L[m][n]

def calculate_similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    
    if not words1 or not words2:
        return 0
    
    lcs_length = lcs(words1, words2)
    max_length = max(len(words1), len(words2))
    
    if max_length == 0:
        return 0
    
    similarity_percentage = (lcs_length / max_length) * 100
    return similarity_percentage

def cosine_similarity_percentage(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_percentage = cosine_sim[0][0] * 100
    return similarity_percentage

def structural_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    paragraphs1 = segment_text(text1)
    paragraphs2 = segment_text(text2)
    
    matched_similarities = []
    for para1 in paragraphs1:
        if not para1.strip():
            continue
        best_match_similarity = 0
        for para2 in paragraphs2:
            if not para2.strip():
                continue
            lcs_sim = calculate_similarity(para1, para2)
            cosine_sim = cosine_similarity_percentage(para1, para2)
            best_match_similarity = max(best_match_similarity, lcs_sim, cosine_sim)
        matched_similarities.append(best_match_similarity)
    
    if not matched_similarities:
        return 0
    
    total_similarity_percentage = sum(matched_similarities) / len(matched_similarities)
    return total_similarity_percentage
def detect_inspirations(text1, text2):
    inspirations = []
    paragraphs1 = segment_text(text1)
    paragraphs2 = segment_text(text2)
    
    for para2 in paragraphs2:  # Iterate through paragraphs in text2
        best_match_similarity = 0
        for para1 in paragraphs1:  # Compare with paragraphs in text1
            lcs_sim = calculate_similarity(para1, para2)
            cosine_sim = cosine_similarity_percentage(para1, para2)
            max_similarity = max(lcs_sim, cosine_sim)
            
            if max_similarity > 0:  # Consider all similarities
                best_match_similarity = max(best_match_similarity, max_similarity)
        inspirations.append((para2, best_match_similarity))
    return inspirations

def calculate_total_inspiration(inspirations, total_paragraphs):
    total_similarity_score = sum(similarity for _, similarity in inspirations)
    if total_paragraphs == 0:
        return 0
    total_inspiration_percentage = total_similarity_score / total_paragraphs
    return total_inspiration_percentage

def measure_inspiration(text1, text2):
    inspirations = detect_inspirations(text1, text2)
    total_inspiration_percentage = calculate_total_inspiration(inspirations, len(inspirations))
    return total_inspiration_percentage

if __name__ == '__main__':
    app.run(debug=True)
