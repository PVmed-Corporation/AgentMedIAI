import LLM
import json
import subprocess
import glob
import os

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
    print(command)
    subprocess.run(
        command,
        text=True
    )

    result_files = [f for f in glob.glob("/data/result/yilinyou/CXASresult/*.npy") ]
    result_files = result_files[0]
    print(result_files)

    cmd = ['/opt/conda/bin/conda', 'run', '-n', 'CXAS', '/opt/conda/envs/CXAS/bin/python', 
               '/data/result/yilinyou/AI agent/CXAS.py',
               '-input', f'{input}',
               '-label', result_files,
               '-output', f'{output}',
               '-target',f'{target}']
    subprocess.run(
        cmd,
        text=True
    )
    os.remove(result_files)

def detect(input, output):
    command = ['/opt/conda/bin/conda', 'run', '-n', 'detection_env', '/opt/conda/envs/detection_env/bin/python', 
               '/data/result/yilinyou/AI agent/detection.py',
               '-input_path', f'{input}',
               '-output_path', f'{output}']
    # print(command)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return stdout,stderr

def agent():
    user_input = "Please help me separate the heart from the image. The image_path is /data/result/yilinyou/00000001_000.png" #input("please input:")
    # print(use_input)
    data = LLM.LLm(user_input)
    # print(data)
    tool = match_tool(data["task"])
    # print(tool["tool_name"])

    if tool["task"]=="segmentation":
        segment(data["input_path"], data["output_path"], data["target_organ"])
    elif tool["task"]=="detection":
        stdout,stderr = detect(data["input_path"], data["output_path"])
        print(stdout)
        print(stderr)
    else:
        print(1)

if __name__ == "__main__":
    agent()