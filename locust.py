import time 
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 5)  # Wait time between tasks

    @task
    def get_homepage(self):
        self.client.get("/")
        time.sleep(1)  # Simulate a delay between requests

    @task
    def get_result(self):
        self.client.get("result/")
        time.sleep(1)  # Simulate a delay between requests