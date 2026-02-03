from datetime import datetime
import os
import io
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import numpy as np
from google.cloud import vision

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
        max_area = 0.0
        max_approx = None
        h_img,w_img = img.shape[:2]
        print("apporx",len(approx))
        print("area",area)
        print("imagearea",image_area * 0.02)
        print("width",w_img)
        print("height",h_img)
        # if len(approx) == 4 and area > (image_area * 0.2):
        hull = cv2.convexHull(c)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if cv2.contourArea(hull) > image_area * 0.02:
            if max_area < area:
                max_area = area
                max_approx = approx
        
        if max_area != 0.0 and cv2.isContourConvex(max_approx):
                x,y,w,h = cv2.boundingRect(max_approx)
                min_x,max_x = x,x+w
                min_y,max_y = y,y+h

                
                # Don't want the one with the page overflows out of the frame
                if max_x < w_img - 5  and max_y < h_img - 5 and min_x > 5 and min_y > 5:
                    print(f"Document detected! Area: {area} max_x: {max_x} max_y: {max_y}")
                    return True, area
        return False, area
                
    return False, 0

@app.route('/ocr', methods=['POST'])
def process_ocr():
    # 1. ファイルがリクエストに含まれているかチェック
    if 'imagefile' not in request.files:
        return jsonify({"status": "error", "message": "No image part"}), 400
    
    file = request.files['imagefile']

    distance_m = request.form.get('distance')
    
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
    
    is_detected, detected_area = is_document_present(filepath)

    if not is_detected:
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

        client = vision.ImageAnnotatorClient.from_service_account_json('gcp.json')
        

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG') # Save to memory buffer
        content = img_byte_arr.getvalue() 

        # 2. Vision API 形式に変換
        image = vision.Image(content=content)

        # 3. テキスト検出を実行
        # document_text_detection は段落やブロック構造の認識に優れています
        response = client.document_text_detection(image=image)

        # エラーチェック
        if response.error.message:
            return jsonify({"error": response.error.message}), 500

        # 4. 結果の抽出
        # texts[0].description には画像内の全テキストが含まれます
        texts = response.text_annotations
        extracted_text = texts[0].description if texts else "テキストは見つかりませんでした"

        result = {
            "status": "success",
            "saved_path": filepath,
            "detected_text": extracted_text, #full_text[:50], # 将来のOCR結果
            "detected_area": detected_area,
            "distance_m": distance_m
        }

        return jsonify(result), 200
        # TODO Third Party OCR API
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    # 3. モックのレスポンス（後にGCP Vision APIをここに統合）
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
