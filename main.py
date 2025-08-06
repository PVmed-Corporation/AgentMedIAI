import LLM
import json
import subprocess
import glob
import os
import computer
import computation as cp
import re

with open('tools.json') as f:
    tools = json.load(f)

def match_tool(task_name):
    return tools.get(task_name)

def segment(input, output, target):
    command = ['/opt/conda/bin/conda', 'run', '-n', 'CXAS', 
               'cxas_segment',
               '-i', f'{input}',
               '-o', '/data/result/yilinyou/CXASresult/',
               '-ot', 'npy']
    # print(command)
    subprocess.run(
        command,
        text=True
    )
    result_files = [f for f in glob.glob("/data/result/yilinyou/CXASresult/*.npy") ]
    result_files = result_files[0]
    # print(result_files)
    spacing = [0.143, 0.143]    # cp.read_spacing(input_csv, input_name)
    
    cmd = ['/opt/conda/bin/conda', 'run', '-n', 'CXAS', '/opt/conda/envs/CXAS/bin/python', 
               '/data/result/yilinyou/AIagent/CXAS.py',
               '-input', f'{input}',
               '-label', result_files,
               '-output', f'{output}',
               '-target',f'{target}']
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, _ = process.communicate()
    # print(stdout)
    lst = [int(x.replace(']', '')) for x in stdout.strip('[]').split()]  
    area = cp.area(lst, result_files, spacing, unit="mm²")
    os.remove(result_files)
    return area

def classify(input):
    command = ['/opt/conda/bin/conda', 'run', '-n', 'classification', '/opt/conda/envs/classification/bin/python', 
               '/data/result/yilinyou/AIagent/classification.py',
               '-input_path', f'{input}']
    # print(' '.join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return stdout,stderr

def chest_detect(input):
    command = ['/opt/conda/bin/conda', 'run', '-n', 'detection_env', '/opt/conda/envs/detection_env/bin/python', 
               '/data/result/yilinyou/AIagent/detection.py',
               '-input_path', f'{input}']
    # print(command)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return stdout,stderr

def Medagent():
    user_input = "分别分割上中下肺。 The input_path is '/data/result/yilinyou/00000001_000.png'"#input("please input:")
    # print(user_input)
    data = LLM.LLm(user_input)
    print(data)
    # input_csv = input("please input metadata_path")
    # input_name = input("please input image_name")

    spacing = [0.143, 0.143]# cp.read_spacing(input_csv, input_name)
    computer_dic = {}
    computer_dic["user_input"] = f"{user_input}"
    computer_dic["spacing"] = {
        "spacing":f"{spacing}",
        "mean":"pixel_spacing"
    }
    for task_key in data:
        # print(task_key)
        task_info = data[task_key]
        # print(task_info)
        tool = match_tool(task_info["task"])
        # print(tool["tool_name"])

        if tool["task"]=="segmentation":
            area = segment(task_info["input_path"], task_info["output_path"], task_info["target_organ"])
            computer_dic["area"] = {
                "area":f"{area}",
                "mean":"The total area of the target region mask image"
            }
            # str_after_remove = task_info["input_path"][:-4]
            # new_chars = "_" + f'{task_info["target_organ"]}' + ".png"
            # result_str = str_after_remove + new_chars
            # print(result_str)
            # computer_dic["segment_image"] = {
            #     "image":f"{result_str}",
            #     "mean":"Visualized images after segmentation"
            # }
        elif tool["task"]=="classification":
            disease, _ = classify(task_info["input_path"])
            print(disease)
        elif tool["task"]=="chest detection":
            stdout, _ = chest_detect(task_info["input_path"])
            print(stdout)

            computer_dic["coordinates1"] = {
                "coordinates":"",
                "mean":"the coordinates of the carina"
            }
            pattern = r"(Carina): \(([\d.]+), ([\d.]+)\)"
            matches = re.findall(pattern, stdout)

            for x, y in matches:
                computer_dic["coordinates1"]["coordinates"] = (float(x), float(y))

            computer_dic["coordinates2"] = {
                "coordinates":"",
                "mean":"the coordinates of the ETT"
            }

            pattern = r"(ETT): \(([\d.]+), ([\d.]+)\)"
            matches = re.findall(pattern, stdout)

            for x, y in matches:
                computer_dic["coordinates2"]["coordinates"] = (float(x), float(y))
        elif tool["task"]=="computation":
            result = computer.computer(f"{computer_dic}")
            print(result)
            # print(computer_dic)
        else:
            print("No corresponding tools")
    # print(computer_dic)
    

if __name__ == "__main__":
    Medagent()