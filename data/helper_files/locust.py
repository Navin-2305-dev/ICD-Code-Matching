import time 
from data.temporary_files.locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def get_homepage(self):
        self.client.get("/")
        time.sleep(1)

    @task
    def get_result(self):
        self.client.get("result/")
        time.sleep(1)