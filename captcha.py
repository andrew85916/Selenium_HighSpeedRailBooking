import requests
import numpy as np
import cv2
from bs4 import BeautifulSoup
from PIL import Image
from image_process import predict_file_preprocessing
from tensorflow.python.keras.models import load_model




session = requests.session()
#使用別人的model
model = load_model("model/thsrc_cnn_model.hdf5")
#驗證碼出現的數字字母
allowedChars = '234579ACFHKMNPQRTYZ'

captcha_file = 'fig/captcha.jpg'
predict_file = 'fig/predict_data.jpg'

def predict_data_process():
    #下載驗證碼
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) ' 'Chrome/84.0.4147.105 Safari/537.36'}
    response = session.get('https://irs.thsrc.com.tw/IMINT/', headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    imgURL="https://irs.thsrc.com.tw"+soup.find('img',id="BookingS1Form_homeCaptcha_passCode")['src']

    try:
        html = requests.get(imgURL, headers=headers)
        with open(captcha_file, 'wb') as f:
            f.write(html.content)
    except ConnectionError:
        print('ConnectionError')
    img = Image.open(captcha_file)
    img = img.resize((140, 48), Image.ANTIALIAS) #重新條調大小，採樣濾波
    img.save(captcha_file, "JPEG")

    #驗證碼處理
    predict_file_preprocessing(captcha_file, predict_file)
    predict_img=cv2.imread(predict_file)

    normalize_predict_img = np.stack([np.array(predict_img) / 255.0])
    prediction = model.predict(normalize_predict_img)
    print(prediction)
    predict_captcha = ''
    for predict in prediction:
        value = np.argmax(predict[0])
        predict_captcha += allowedChars[value]
    return predict_captcha


