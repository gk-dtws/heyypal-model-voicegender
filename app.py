from flask import Flask, request, jsonify
from flask_cors import CORS

from werkzeug.utils import secure_filename
import os
from utils import create_model, extract_feature, is_valid_wav, transcribe_audio, verify_transcription

app = Flask(__name__)
CORS(app)
model = create_model()
model.load_weights("models/model.h5")

UPLOAD_FOLDER = "uploads"
TEXT = "Hi this is an example text for getting similarity score."
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#Home page
@app.route("/", methods=["GET"])
def home():
    return "🎙️ Gender Prediction API is up. Use POST /predict to analyze voice."

@app.route("/text", methods=["GET"])
def display_text():
    return jsonify({"text": TEXT}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe_and_verify():
    print(request, request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    # If file name is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(file_path)

        if not is_valid_wav(file_path):
            return jsonify({'error': 'Invalid WAV file. Please upload a valid 16-bit WAV audio file.'}), 400

        transctiption = transcribe_audio(file_path)
        if transctiption is None:
            return jsonify({'error': 'Failed to transcribe audio.'}), 400
        
        return jsonify({
            'transcription': transctiption,
            'similarity': verify_transcription(transctiption, TEXT)
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


#/predict post request
@app.route("/predict", methods=["POST"])
def predict():
    # If no file is attached to the request
    print(request, request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    # If file name is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(file_path)

        if not is_valid_wav(file_path):
            return jsonify({'error': 'Invalid WAV file. Please upload a valid 16-bit WAV audio file.'}), 400

        features = extract_feature(file_path, mel=True).reshape(1, -1)
        if features is None:
            return jsonify({'error': 'Failed to extract features from audio.'}), 400

        # Making prediction
        prediction = model.predict(features, verbose=0)
        male_prob = prediction[0][0]
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        confidence = float(max(male_prob, female_prob))

        return jsonify({
            'gender': gender,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
