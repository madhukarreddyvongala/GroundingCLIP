import openai
import json
import os
import pandas as pd
from tqdm import tqdm
import re

openai.api_key = "" # Add your OpenAI API key here


# base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets/Flickr"))
# data_path = os.path.join(base_dir, "flickr_test_top10.csv")
# SAVE_DIR = os.path.join(base_dir, "Flickr_relations")


base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets/MSCOCO"))
data_path = os.path.join(base_dir, "Coco_test_top10.csv")
SAVE_DIR = os.path.join(base_dir, "Coco_relations")


data = pd.read_csv(data_path)
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_TEXT_JSON_PATH = os.path.join(SAVE_DIR, "{}.json")

def get_relation(text):
    input = (
        f'Given the caption """{text}""", identify all the objects mentioned, their attributes (if any), '
        'and describe how these objects are connected using subject-verb-object relationships. '
        'Return the output in the following JSON format:\n\n'
        '{\n'
        '  "objects": {\n'
        '    "object1": {"attributes": "attribute1 or null"},\n'
        '    "object2": {"attributes": "attribute2 or null"}\n'
        '  },\n'
        '  "connections": [\n'
        '    {"subject": "object1", "verb": "verb", "object": "object2"}\n'
        '  ]\n'
        '}\n\n'
        'Examples:\n\n'
        'Caption: """young person sits on a boat"""\n'
        'Output:\n'
        '{\n'
        '  "objects": {\n'
        '    "person": {"attributes": "young"},\n'
        '    "boat": {"attributes": null}\n'
        '  },\n'
        '  "connections": [\n'
        '    {"subject": "person", "verb": "sits", "object": "boat"}\n'
        '  ]\n'
        '}\n\n'
        'Caption: """a dog is lying under a wooden table"""\n'
        'Output:\n'
        '{\n'
        '  "objects": {\n'
        '    "dog": {"attributes": null},\n'
        '    "table": {"attributes": "wooden"}\n'
        '  },\n'
        '  "connections": [\n'
        '    {"subject": "dog", "verb": "lying under", "object": "table"}\n'
        '  ]\n'
        '}\n\n'
        'Caption: """several people walking past a food cart on a city street"""\n'
        'Output:\n'
        '{\n'
        '  "objects": {\n'
        '    "people": {"attributes": "several"},\n'
        '    "food cart": {"attributes": null},\n'
        '    "city street": {"attributes": null}\n'
        '  },\n'
        '  "connections": [\n'
        '    {"subject": "people", "verb": "walking past", "object": "food cart"},\n'
        '    {"subject": "food cart", "verb": "on", "object": "city street"}\n'
        '  ]\n'
        '}\n\n'
        'Caption: """a wall is being painted by two workers"""\n'
        'Output:\n'
        '{\n'
        '  "objects": {\n'
        '    "wall": {"attributes": null},\n'
        '    "workers": {"attributes": "two"}\n'
        '  },\n'
        '  "connections": [\n'
        '    {"subject": "workers", "verb": "painting", "object": "wall"}\n'
        '  ]\n'
        '}\n\n'
        'Note: If an object is a descriptive feature, article of clothing, or accessory attached to or worn by another object (e.g., "orange hat" on a "man"), include it as an attribute of that object, unless it acts independently.\n'
        'Note: Do not include vague references like "something", "someone", or "anything" as objects in the output. However, you can still use them in the connections if they appear in the caption.\n'

        'Just output the json exactly like the format in example.'
    )

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input}],
        temperature=0
    )
    return completion.choices[0].message.content



def extract_json_from_string(s):
    match = re.search(r'{.*}', s, re.DOTALL)
    return match.group() if match else None

if __name__ == "__main__":
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing captions"):

        if idx == 10:
            break

        save_path = row['image_path'].split("/")[-1].split(".")[0]
        output_path = SAVE_TEXT_JSON_PATH.format(save_path)

        if not os.path.exists(output_path):
            result = get_relation(row.caption)

            if isinstance(result, str):
                result_str = extract_json_from_string(result)
                if result_str is None:
                    print(f"Failed to extract JSON for row {idx}:\n{result}")
                    continue
                try:
                    result = json.loads(result_str)
                except json.JSONDecodeError:
                    print(f"JSON parsing error at row {idx}:\n{result_str}")
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    continue

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)




               

