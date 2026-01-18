from datetime import datetime
import mediapipe as mp
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np

app = Flask(__name__)

# 保存先ディレクトリの設定
UPLOAD_FOLDER = 'received_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_white_document(img, contour):
    """
    指定された輪郭の内部が『白（書類）』に近いか判定する
    """
    # 1. 輪郭に基づいたマスクを作成
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # 2. 画像をHSV形式に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. マスク範囲内（書類の内側）の平均値を計算
    # mean[0]: Hue, mean[1]: Saturation, mean[2]: Value
    mean = cv2.mean(hsv, mask=mask)
    
    # 4. 判定基準（2026年現在の環境に合わせた標準値）
    # Saturation（彩度）が低い = 白・灰・黒に近い
    # Value（明度）が高い = 白に近い
    is_low_saturation = mean[1] < 50  # 0~255。低いほど色が薄い
    is_high_value = mean[2] > 160     # 0~255。高いほど明るい（白い）

    is_low_saturation = mean[1] < 110  # 50から110に緩和（多少色が付いていても許容）
    is_high_value = mean[2] > 130 

    print(f"Color Check - Saturation: {mean[1]:.1f}, Value: {mean[2]:.1f}")
    
    return is_low_saturation and is_high_value


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
        if len(approx) == 4 and cv2.contourArea(hull) > image_area * 0.02:
            # 凸包チェック（凹んでいる四角形を除外）
            if cv2.isContourConvex(approx):
                # --- 追加：色のチェック ---
                if is_white_document(img, hull):
                    print("Document detected! (Shape + Color match)")
                    return True
                else:
                    print("Shape matched, but color is not white enough.")
                    
                # print(f"Document detected! Area: {area}")
                # return True
                
    return False

@app.route('/ocr', methods=['POST'])
def process_ocr():
    if 'imagefile' not in request.files:
        return jsonify({"status": "error", "message": "No file"}), 400
    
    file = request.files['imagefile']
    
    # 一旦一時保存
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_check.jpg")
    file.save(temp_path)

    # 書類があるか判定
    if is_document_present(temp_path):
        # 書類があればタイムスタンプを付けて正式保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"doc_{timestamp}.jpg"
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        os.rename(temp_path, final_path)
        
        # TODO: ここで Google Cloud Vision API を呼び出す
        return jsonify({"status": "success", "message": "Document found", "path": final_path}), 200
    else:
        # 書類がなければ削除
        # os.remove(temp_path)
        return jsonify({"status": "ignored", "message": "No document detected"}), 202

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
