from datetime import datetime
import os
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import numpy as np

app = Flask(__name__)

# 保存先ディレクトリの設定
UPLOAD_FOLDER = 'received_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_blur_score(image_path):
    """
    Returns the variance of the Laplacian. 
    High score = Sharp (Edges are clear)
    Low score = Blurry (Edges are smoothed out)
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Variance of the Laplacian is the industry standard for blur detection
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def is_document_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # 1. 前処理：グレースケール化とぼかしでノイズ除去
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. エッジ検出（Canny）
    edged = cv2.Canny(blurred, 30, 150) # しきい値を少し下げて拾いやすくする

    # Cannyの後に「膨張(Dilation)」処理を追加
    edged = cv2.Canny(blurred, 30, 150)
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1) # エッジを太くする


    # 3. 輪郭抽出
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    image_area = img.shape[0] * img.shape[1]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 判定条件：
        # - 頂点が4つ（四角形）
        # - 面積が画像全体の20%以上（小さすぎるノイズを除外）
        area = cv2.contourArea(c)
        print("apporx",len(approx))
        print("area",area)
        print("imagearea",image_area * 0.02)
        # if len(approx) == 4 and area > (image_area * 0.2):
        hull = cv2.convexHull(c)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if cv2.contourArea(hull) > image_area * 0.02:
            # 凸包チェック（凹んでいる四角形を除外）
            if cv2.isContourConvex(approx):
                print(f"Document detected! Area: {area}")
                return True
                
    return False

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

    # 1. Blur Check
    blur_score = get_blur_score(filepath)
    print(f"Blur Score: {blur_score:.2f}")

    # Threshold varies by lighting, but 100 is a good starting point
    if blur_score < 60:
        print("❌ Image too blurry. Deleting...")
        os.remove(filepath)
        return jsonify({"status": "error", "message": "Image too blurry"}), 400
    
    if is_document_present(filepath) == False:
        os.remove(filepath)
        return jsonify({"status": "error", "message": "Document is not presented"}), 400

    print(f"Saved image to: {filepath}")


    try:
        # 画像を読み込み
        img = Image.open(filepath)
        
        # --- 改善1: 言語とPSMを指定して精度を上げる ---
        # ドイツ語の本であれば lang='deu' を指定
        data = pytesseract.image_to_data(img, lang='eng', config='--psm 3', output_type=pytesseract.Output.DICT)
        
        # 全単語の信頼度の平均を計算
        confidences = [int(c) for c in data['conf'] if int(c) != -1]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        # --- 改善2: 空の結果をハンドリング ---
        # 信頼度計算ができたとしても、有効なテキストが一つも無い場合があります
        text_list = [t.strip() for t in data['text'] if t.strip()]
        full_text = " ".join(text_list)

        if avg_conf < 30 or len(full_text) < 3: 
            print(f"Skipping gibberish (Conf: {avg_conf:.1f}%, Text Length: {len(full_text)})")
            # 失敗した画像はサーバーの容量を食うので削除しても良い
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"status": "error", "message": "Low OCR confidence or no text"}), 400

        # 成功した場合の処理
        print(f"OCR Success (Conf: {avg_conf:.1f}%): {full_text[:50]}...")
        # TODO Third Party OCR API
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    # 3. モックのレスポンス（後にGCP Vision APIをここに統合）
    mock_result = {
        "status": "success",
        "saved_path": filepath,
        "detected_text": "Hello World 2026", # 将来のOCR結果
    }

    return jsonify(mock_result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
