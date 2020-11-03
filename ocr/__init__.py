import cv2
import requests
import os
import matplotlib.pyplot as plt
import image_to_numpy
import numpy as np

LIMIT_PX = 1024
LIMIT_BYTE = 1024 * 1024  # 1MB
LIMIT_BOX = 40

API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'


def kakao_ocr_resize(image_path: str):
    """
    ocr detect/recognize api helper
    ocr api의 제약사항이 넘어서는 이미지는 요청 이전에 전처리가 필요.

    pixel 제약사항 초과: resize
    용량 제약사항 초과  : 다른 포맷으로 압축, 이미지 분할 등의 처리 필요. (예제에서 제공하지 않음)

    :param image_path: 이미지파일 경로
    :return:
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    if LIMIT_PX < height or LIMIT_PX < width:
        ratio = float(LIMIT_PX) / max(height, width)
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        height, width, _ = height, width, _ = image.shape

        # api 사용전에 이미지가 resize된 경우, recognize시 resize된 결과를 사용해야함.
        image_path = "{}_resized.jpg".format(image_path)
        cv2.imwrite(image_path, image)

        return image_path
    return None


def kakao_ocr(image_path: str, api_key: str):
    """
    OCR api request example
    :param image_path: 이미지파일 경로
    :param api_key: 카카오 앱 REST API 키
    """
    resize_impath = kakao_ocr_resize(image_path)
    if resize_impath is not None:
        image_path = resize_impath

    headers = {'Authorization': 'KakaoAK {}'.format(api_key)}

    image = image_to_numpy.load_image_file(image_path)
    image = cv2.imencode('.png', image)[1]
    data = image.tobytes()
    api_response = requests.post(API_URL, headers=headers, files={"image": data}).json()

    print(api_response)

    if resize_impath is not None:
        os.remove(resize_impath)

    if len(api_response['result']) == 0:
        return 'No words found'

    result = [token['recognition_words'][0] for token in api_response['result']]

    return ' '.join(result)


def main():
    # image_path = 'C:/Users/Sopiro/PycharmProjects/flask_app/static/uploads/2020-10-31_213104.jpg'
    image_path = 'C:/Users/Sopiro/test.jpg'

    output = kakao_ocr(image_path, '1b9ef11c3bdeaa8cb71013c0e2ecb9f9')

    print(output)


if __name__ == '__main__':
    main()
