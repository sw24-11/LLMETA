from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append('./ai/vision/Deeper_RelTR/')

from llm.text_inference import llama2_chain, extract_info
from vision.Deeper_RelTR.img_inference import vision_inference, get_args_parser, argparse

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_inference(research_paper):
    model_paths = ["C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf"]
    n_batches = [2000, 3000, 4000]
    n_gpu = 50

    llm = llama2_chain(model_paths[0], n_batch=n_batches[0], n_gpu_layers=n_gpu, input_paper=research_paper)
    llm_chain, prompt = llm.llm_set()
    response = llm_chain.invoke(prompt)
    info_dict = extract_info(response['text'])
    print(info_dict)
    return info_dict

def image_inference(img_path):
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser(img_path=img_path)])
    args = parser.parse_args()
    caption, triplet_graph = vision_inference(args)
    return triplet_graph

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'data_type' not in data:
        return jsonify({'error': 'Invalid input data'}), 400

    data_type = data['data_type']
    if data_type == 'text':
        text = data.get('text')
        if text:
            metadata = text_inference(text)
            return jsonify({'metadata': metadata})
        else:
            return jsonify({'error': 'Text data not provided'}), 400

    elif data_type == 'image':
        img_path = data.get('image_path')
        if img_path and allowed_file(img_path):
            if not os.path.isfile(img_path):
                return jsonify({'error': 'Image file not found'}), 400

            metadata = image_inference(img_path)
            return jsonify({'metadata': metadata})
        else:
            return jsonify({'error': 'Image path not provided or invalid'}), 400

    else:
        return jsonify({'error': 'Invalid data_type'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
