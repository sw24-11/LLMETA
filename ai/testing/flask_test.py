from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import sys
import fitz  # PyMuPDF

sys.path.append('./ai/vision/Deeper_RelTR/')

from llm.text_inference import llama2_chain, extract_info
from vision.Deeper_RelTR.img_inference import vision_inference, get_args_parser, argparse

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def truncate_text(text, max_length=3072):
    # Split the text into words and truncate to the maximum length allowed
    words = text.split()
    truncated_text = ' '.join(words[:max_length])
    return truncated_text

import re

def extract_before_introduction(text):
    abstract_pattern = re.compile(r'abstract', re.IGNORECASE)
    introduction_pattern = re.compile(r'introduction', re.IGNORECASE)
    reference_pattern = re.compile(r'reference', re.IGNORECASE)

    abstract_match = abstract_pattern.search(text)
    if not abstract_match:
        return "Abstract section not found"

    introduction_match = introduction_pattern.search(text, abstract_match.end())
    if not introduction_match:
        return text[0:abstract_match.start()]

    return text[0:introduction_match.start()]

def text_inference(research_paper):
    model_paths = ["C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf"]
    n_batches = [2000, 3000, 4000]
    n_gpu = 50

    # Truncate text to fit within the model's token limit
    truncated_paper = extract_before_introduction(research_paper)

    llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu,
                       input_paper=truncated_paper, cb_manager=0)
    llm_chain, prompt = llm.llm_set()
    response = llm_chain.invoke(prompt)
    info_dict = extract_info(response['text'])
    return info_dict

def image_inference(img_path):
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser(img_path=img_path)])
    args = parser.parse_args()
    caption, triplet_graph = vision_inference(args)
    return triplet_graph

def extract_text_from_pdf(file_path):
    # Extract text from the uploaded PDF file using PyMuPDF
    document = fitz.open(file_path)
    text = ''
    for page in document:
        text += page.get_text("text")
    return text

@app.route('/')
def home():
    return render_template('test1.html')

import time
#/metadata/extraction
@app.route('/metadata/extraction', methods=['POST'])
def analyze_text():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400
    file = request.files['pdf']
    if file and allowed_file(file.filename):
        print(time.time())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(time.time())

        try:
            print(time.time())
            text_content = extract_text_from_pdf(file_path)
            print(text_content)
            metadata = text_inference(text_content)
            print(time.time())
            return jsonify({'metadata': metadata})

        finally:
            os.remove(file_path)

    return jsonify({'error': 'Invalid file type or file not uploaded properly'}), 400

@app.route('/metadata/extraction-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            metadata = image_inference(file_path)

            return jsonify({'metadata': metadata})

        finally:
            os.remove(file_path)

    return jsonify({'error': 'Invalid file type or file not uploaded properly'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
