import cv2
# numpyは画像データの格納
import numpy as np


def camera():
    #カメラ映像の読み込み
    cap = cv2.VideoCapture(0)
    #動画ファイルの初期化の確認
    isOpened = cap.isOpened()
    if not isOpened:
        return
    while True:
        result, frame = cap.read()
        if not result:
            return
        # 画像表示
        cv2.imshow('camera', frame)
        # キー入力受付(30ms)
        key = cv2.waitKey(30)
        # 終了キー（EnterかEscで終了）
        if (key == 13) or (key == 27):
            break
    # カメラ終了
    cap.release()
    cv2.destroyAllWindows()

camera()