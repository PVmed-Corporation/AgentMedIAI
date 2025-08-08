from openai import OpenAI
import json

with open('key.json') as f:
    key = json.load(f)

def computer(user_input):
    client = OpenAI(
        api_key=key["api_key"],
        base_url="https://api.deepseek.com",
    )

    system_prompt = """
    You are a calculator. Users will input a dictionary.
    The dictionary input contains values and definitions. The 'mean' is the values' definitions.
    You need to derive the formula based on the input and calculate the result by combining the given parameters.
    Or the user may provide you with the path of the processed image, and you will perform the calculation based on the image provided by the user.

    Example user_input:
    "{
    'user_input': "Please help me calculate the distance between Beijing and Shanghai on the map.", 
    'coordinates1': {'coordinates': "(243, 268)", 'mean': "the coordinates of the Beijing"}, 
    'coordinates2': {'coordinates': "(687, 728)", 'mean': "the coordinates of the Shanghai"}, 
    'Pantograph ratio': {'Pantograph ratio': "1:100000", 'mean': "the Pantograph ratio of map"}
    }"

    Example output:
    result = ....
    """

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return response.choices[0].message.content