import json
from openai import OpenAI

with open('key.json') as f:
    key = json.load(f)

with open('label.json') as f:
    label = json.load(f)

def LLm(user_input):
    client = OpenAI(
        api_key = key["api_key"],
        base_url="https://api.deepseek.com",
    )

    system_prompt = """
    You are now the planner of a medical image processing agent. 
    Users will input their requirements for medical image processing to you, 
    and you need to plan and sort the process of image processing based on their requirements and output them in json format. 
    The tasks of the plan can only be selected from segmentation, classification, chest detection, computation, unknown task.
    And all the output paths are "/data/result/yilinyou/result/".
    If the user input contains multiple tasks, please split them into multiple sub-tasks.
    For the segmentation task, the target_organ can only come from """ + f"{label}" +\
    """
    For classification tasks, only the following several diseases can be classified.
        Atelectasis,
        Consolidation,
        Infiltration,
        Pneumothorax,
        Edema,
        Emphysema,
        Fibrosis,
        Effusion,
        Pneumonia,
        Pleural_Thickening,
        Cardiomegaly,
        Nodule,
        Mass,
        Hernia,
        Lung Lesion,
        Fracture,
        Lung Opacity,
        Enlarged Cardiomediastinum.
    For the chest detection task, only the coordinates of the tracheal intubation and the carina can be calculated..
    If there are no relevant tasks to choose from, you can return to the unknown task.
    The computation task is always carried out after all other tasks.
    For computational tasks, you must provide the target that the user wants to calculate based on their input.
    The calculation targets can only be selected from these four options: distance, area, percentage, and rate of change.

    EXAMPLE JSON OUTPUT:
    {
        "task1":{
        "task":"segmentation",
        "input_path":"path_to_read_image",
        "output_path":"path_to_save_image",
        "target_organ":"target_organ"
    }
    {
        "task1":{
        "task":"classification",
        "input_path":"path_to_read_image"
    }
    {
        "task1":{
        "task":"chest detection",
        "input_path":"path_to_read_image"
    }
    {
        "task1":{
        "task":"computation",
        "computation_target":"user_target"
    }
    {
        "task1":{
        "task":"unknown task"
    }
    For multiple tasks example:
    {
        task1:{
            "task":"segmentation",
            "input_path":"path_to_read_image",
            "output_path":"path_to_save_image",
            "target_organ":"target_organ"
        },
        task2:{
            "task":"classification",
            "input_path":"path_to_read_image"
        }
    }
    For the multi-tasking involves unknown tasks example:
    {
        task1:{
            "task":"segmentation",
            "input_path":"path_to_read_image",
            "output_path":"path_to_save_image",
            "target_organ":"target_organ"
        },
        task2:{
            "task":"unknown task"
        }
    }
    """
    # print(system_prompt)
    # user_prompt = "Can you analyze this chest CT for lung nodules and give me their size and location?"

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    # print(json.loads(response.choices[0].message.content))
    return json.loads(response.choices[0].message.content)