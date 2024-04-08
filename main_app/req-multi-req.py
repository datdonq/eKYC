import aiohttp
import asyncio
from aiohttp import FormData

# Định nghĩa hàm gửi yêu cầu
async def send_request(session, url, image_path, type_card, is_front):
    data = FormData()
    data.add_field('image', open(image_path, 'rb'), filename='image.jpg')
    data.add_field('type_card', str(type_card))
    data.add_field('is_front', str(is_front))

    async with session.post(url, data=data) as response:
        return await response.text()

# Định nghĩa hàm chính để tạo và gửi 20 yêu cầu cùng lúc
async def main():
    url = "http://localhost:8008/api/v1/idcardreader"  # Sửa đường dẫn URL thật
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(20):
            # Sửa các tham số image_path, type_card, is_front tương ứng
            tasks.append(send_request(session, url, './assets/front-cccd.jpg', 0, 1))
        responses = await asyncio.gather(*tasks)
        for response in responses:
            print(response)

# Chạy hàm chính
asyncio.run(main())
