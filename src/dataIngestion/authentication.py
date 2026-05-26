from dotenv import load_dotenv
from huggingface_hub import login
import os


def hf_login():

    load_dotenv()

    token = os.getenv("HF_TOKEN")

    if token is None:
        raise ValueError("HF_TOKEN not found in .env")

    login(token=token)