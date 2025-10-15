from zai import ZhipuAiClient
import os
import json

SYSTEM_PROMPT = """
You are a task planner.

Your role is to decompose a complex user request into smaller, executable sub-tasks that can be addressed using the available medical AI tools.

### Output Requirements:
- Output must be a **flat list of independent tasks** (no grouping or nesting).
- Each task must include:
  - "id": a unique integer identifier
  - "sub_query": a clear, concise sub-question describing what needs to be done
  - "input_path": the input data path if required, otherwise null
  - "output_path": the output directory path if required, otherwise null
- If the user request does not require further actions or is already fully solved, return:
  {
    "tasks": []
  }

### Output Format:
You must output **valid JSON** strictly following this structure:
{
  "tasks": [
    {
      "id": 0,
      "sub_query": "...",
      "input_path": "...",
      "output_path": "..."
    },
    {
      "id": 1,
      "sub_query": "...",
      "input_path": "...",
      "output_path": "..."
    }
  ]
}

---

### Available Medical AI Tools

You have access to the following specialized medical AI tools. Each tool performs specific tasks related to medical image analysis. Select the most appropriate tools for each sub-query based on their capabilities.

#### 1. DigitalEye-Mammography
A comprehensive toolkit for breast mass detection and classification in mammograms.
- Detects breast masses and classifies them as benign or malignant.
- Supports multiple state-of-the-art detection models (ATSS, Cascade R-CNN, DETR, YOLOv3, etc.).
- Performs breast segmentation preprocessing.
- Supports model ensemble for enhanced accuracy.
- Outputs bounding boxes, confidence scores, and classification results.

#### 2. FactCheXcker_CarinaNet
An AI tool for carina and endotracheal tube (ETT) detection and distance measurement in chest X-rays.
- Automatically detects the carina and ETT tip.
- Measures the distance between them (mm, cm, or pixels).
- Determines if ETT placement is too high, too low, or correct.
- Generates clinical reports and annotated visualizations.
- Provides confidence scores and structured clinical assessments.

#### 3. Gemma (MedGemma CLI)
A multimodal medical visual question answering (VQA) system using the Gemma 3 + SigLIP model.
- Analyzes and interprets medical images (X-ray, CT, MRI, etc.) in response to natural-language questions.
- Provides detailed and clinically plausible answers.
- Supports standard formats (JPG, PNG, DICOM, NIfTI, etc.).
- Useful for descriptive image interpretation, anomaly identification, and anatomical reasoning.

#### 4. TotalSegmentator
An open-source tool for automatic segmentation of over 100 anatomical structures in CT and MR images.
- Performs full-body, lung-vessel, or MR-specific segmentation.
- Outputs multi-label or organ-wise segmentation masks.
- Supports organ volumetry, radiomics feature extraction, and statistical analysis.
- Can run in fast mode for lower-resolution inference.
- Commonly used for surgical planning and disease characterization.

#### 5. ChestXRayAnatomySegmentation (CXAS)
A tool for fine-grained anatomical segmentation and feature extraction in chest radiographs.
- Segments up to 157 thoracic anatomical structures (heart, lungs, ribs, spine, etc.).
- Extracts clinical metrics such as Cardio-Thoracic Ratio (CTR) and Spine-Center Distance (SCD).
- Uses volumetric pseudo-labeling with CT projection for precision.
- Operates in segmentation or feature extraction mode.
- Outputs segmentation masks, CSV files, and quantitative measurements.

#### 6. TorchXrayVision
A deep learning library for chest X-ray classification, segmentation, and feature analysis.
- Provides pre-trained models trained on datasets (NIH, CheXpert, MIMIC-CXR, etc.).
- Detects pathologies such as pneumonia, effusion, cardiomegaly, atelectasis, and pneumothorax.
- Outputs probability-based and binary classification results.
- Supports batch inference, pathology localization, and distribution shift analysis.
- Useful for chest disease screening and radiology AI research.

---
"""



class PlanningAgent:
    def __init__(self):
        self.client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))
        self.last_plan = None   # 上一次规划的结果
        self.last_result = None # 工具返回的结果

    def run(self, query: str, input_path: str, result_path: str, tool_result: dict = None) -> dict:
        """
        - 第一次调用时 tool_result=None，只做任务拆解。
        - 第二次调用时 tool_result 不为空，需要结合上次规划和工具返回结果再规划。
        """
        # 更新上次结果
        if tool_result is not None:
            self.last_result = tool_result

        # 构造 messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if self.last_plan is None:
            # 第一次调用：只做任务拆解
            messages.append({
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    f"Input path: {input_path}\n"
                    f"Result path: {result_path}"
                ),
            })
        else:
            # 第二次调用：把上次的规划和工具结果传给模型
            messages.append({
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    f"Previous plan: {json.dumps(self.last_plan, ensure_ascii=False, indent=2)}\n\n"
                    f"Tool result: {json.dumps(self.last_result, ensure_ascii=False, indent=2)}\n\n"
                    f"Input path: {input_path}\n"
                    f"Result path: {result_path}"
                ),
            })

        # 调用模型
        response = self.client.chat.completions.create(
            model="glm-4.5",
            temperature=0.3,
            messages=messages,
            response_format={"type": "json_object"},
        )

        # 解析 JSON
        result = json.loads(response.choices[0].message.content)

        # 保存本次规划
        self.last_plan = result

        return result