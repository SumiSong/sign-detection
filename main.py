# 전체 구조
# 웹캠 + 미디어파이프 -> 좌표 추출 -> POST 전송
# Flask 에서 데이터 받음 -> 모델 호출 -> 예측 -> 예측 결과 반환
# {
#     "pose": [50개 값],
#     "left_hand": [42개 값],
#     "right_hand": [42개 값],
#     "face": [140개 값]
# }

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.python.keras.saving.save import load_model

app = Flask(__name__)

# 모델 로드
model = load_model("sign_dnn_model.h5")

# 에측
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # 데이터 받아온 것
        pose = np.array(data['pose']).reshape(1, -1)
        left_hand = np.array(data['left_hand']).reshape(1, -1)
        right_hand = np.array(data['right_hand']).reshape(1, -1)
        face = np.array(data['face']).reshape(1, -1)

        # # 예측
        # predict = model.predict([pose, left_hand, right_hand, face])
        # pred_index = np.argmax(predict)
        # word =
    except Exception as e:
        return jsonify({'error': str(e)})


#설정
if __name__ == '__main__':
    app.run(debug=True)

