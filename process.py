import pyarrow.ipc as ipc
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


SIZE = 300000
SIZE_1 = SIZE * 0.15
SIZE_2 = SIZE * 0.15
SIZE_3 = SIZE * 0.20
SIZE_4 = SIZE * 0.20
SIZE_5 = SIZE * 0.30

def process_dataset():
    file_name = "state.json"

    with open(file_name, "r") as f:
        data = json.load(f)
        
    name = data["_data_files"][0]

    with open(name["filename"], "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
        
    texts = table[1].to_pylist()
    ratings = table[0].to_pylist()

    dict = []
    one_star = 0
    two_star = 0
    three_star = 0
    four_star = 0
    five_star = 0

    for rating, text in zip(ratings, texts):
        if (int(rating) == 1) and (one_star < SIZE_1):
            one_star = one_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        elif (int(rating) == 2) and (two_star < SIZE_2):
            two_star = two_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        elif (int(rating) == 3) and (three_star < SIZE_3):
            three_star = three_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        elif (int(rating) == 4) and (four_star < SIZE_4):
            four_star = four_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        elif (int(rating) == 5) and (five_star < SIZE_5):
            five_star = five_star + 1
            item = {"text": text, "rating": rating}
            dict.append(item)
        elif (one_star == SIZE_1 and two_star == SIZE_2 and three_star == SIZE_3 and four_star == SIZE_4 and five_star == SIZE_5):
            break
    return dict