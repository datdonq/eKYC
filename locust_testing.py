from locust import HttpUser, TaskSet, task
class WebsiteTasks(TaskSet):
    @task()
    def predict_binary(self):
        # Định nghĩa dữ liệu ảnh đầu vào, đây là một ví dụ, bạn cần thay thế bằng dữ liệu thực tế
        image_data = open("Card_Reader/detect_cccd_front/test/test_1.jpg", "rb").read()

        # Gửi yêu cầu POST đến API với ảnh là dữ liệu đầu vào
        response = self.client.post("/predict_binary", files={"binary_file": image_data})

        # Kiểm tra phản hồi từ API
        if response.status_code == 200:
            print("API response:", response.text)
        else:
            print("API response error:", response.status_code)

class WebsiteUser(HttpUser):
    tasks = [WebsiteTasks]
    min_wait = 50
    max_wait = 70
