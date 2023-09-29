from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram import Bot
import cv2
import os


TOKEN_API = "6356861975:AAHeJy29pGeG9W11iXuk27bQn1R8A6dGQ1E"


bot = Bot(TOKEN_API)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help', 'info'])
async def info(message):
    await message.answer("""
Привет! 
Этот бот умеет определять по изображению вид сорняка: бодяк, осот или конский щавель. 
Чтобы воспользоваться ботом, отправьте изображение растения. 
Бот выделит на изображении сорняк с помощью рамки, подпишет его название и укажет, на сколько он уверен в точности его определения.
""")

@dp.message_handler(content_types=["photo"])
async def get_photo(message):
    file_info = await bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('photos/')[1]
    await message.photo[-1].download("static/img/" + file_name)
    image = cv2.imread("static/img/" + file_name)
    from yolo_predictions import YOLO_Pred
    try:

        yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
        img_pred = yolo.predictions(image)
        cv2.imwrite(os.path.join("static/answer", file_name), img_pred)

        await message.answer_photo(photo=open(f'static/answer/{file_name}', 'rb'))
    except AttributeError:
        await message.answer("нет сорняко")


if __name__ == '__main__':
    executor.start_polling(dp)
