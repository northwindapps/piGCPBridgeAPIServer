from datetime import datetime
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 保存先ディレクトリの設定
UPLOAD_FOLDER = 'received_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/ocr', methods=['POST'])
def process_ocr():
    # 1. ファイルがリクエストに含まれているかチェック
    if 'imagefile' not in request.files:
        return jsonify({"status": "error", "message": "No image part"}), 400
    
    file = request.files['imagefile']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    # --- 修正ポイント：ファイル名にタイムスタンプを付与 ---
    # 1. 元のファイル名から拡張子を取得 (例: .jpg)
    ext = os.path.splitext(secure_filename(file.filename))[1]
    
    # 2. 現在時刻からユニークなファイル名を作成 (例: 20260118_131502_123.jpg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    new_filename = f"{timestamp}{ext}"
    
    # 3. 保存パスの生成
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    
    file.save(filepath)

    print(f"Saved image to: {filepath}")

    # 3. モックのレスポンス（後にGCP Vision APIをここに統合）
    mock_result = {
        "status": "success",
        "saved_path": filepath,
        "detected_text": "Hello World 2026", # 将来のOCR結果
    }

    return jsonify(mock_result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
