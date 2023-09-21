import pickle
from utils import OpenAIAsync
import asyncio
import dotenv
import os

dotenv.load_dotenv()

if __name__ == "__main__":
    with open("metar_strings.pickle", 'rb') as pickle_file:
        metar_strings = pickle.load(pickle_file)

    openai_instance = OpenAIAsync(os.getenv("OPENAI_API_KEY"))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(openai_instance.main(metar_strings))