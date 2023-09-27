from io import BytesIO

import cv2
import numpy as np
import os
import telebot
from yolo_predictions import YOLO_Pred


bot = telebot.TeleBot('6356861975:AAHeJy29pGeG9W11iXuk27bQn1R8A6dGQ1E')

@bot.message_handler(commands=['start', 'help'])
def start(message):
    message_text = f'<b>Привет, {message.from_user.first_name}</b>'
    bot.send_message(message.chat.id, message_text, parse_mode='html')


@bot.message_handler(content_types=['photo'])
def get_user_photo(message):

    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    src = file_info.file_path
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(message, "Фото добавлено")


    yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
    img = cv2.imread(src)
    # cv2.imshow("1", img)
    img_pred = yolo.predictions(img)
    cv2.imshow('prediction image', img_pred)

    cv2.imwrite(os.path.join('photos', "Yolo_" + src), img_pred)



bot.polling(none_stop=True)

