from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram import Bot
import cv2
import os

bot = Bot("6356861975:AAHeJy29pGeG9W11iXuk27bQn1R8A6dGQ1E")
dp = Dispatcher(bot)


@dp.message_handler(content_types=["photo"])
async def get_photo(message):
    file_info = await bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('photos/')[1]
    await message.photo[-1].download(file_info.file_path.split('photos/')[1])
    image = cv2.imread(file_name)
    from yolo_predictions import YOLO_Pred
    try:

        yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
        img_pred = yolo.predictions(image)
        cv2.imwrite(os.path.join('photos', "1.jpg"), img_pred)

        await message.answer_photo(photo=open(f'photos/1.jpg', 'rb'))
    except AttributeError:
        await message.answer("нет сорняко")

    # await message.photo[-1].download(file_info.file_path.split('photos/')[1]) # ++





if __name__ == '__main__':
    executor.start_polling(dp)