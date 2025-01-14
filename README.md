# 🚀 **SafeInfer**: Context Adaptive Decoding Time Safety Alignment for Large Language Models  
### 🎉 **Accepted at AAAI-2025 (Long Paper)** — Alignment Track  

Language models aligned for safety often exhibit fragile and imbalanced mechanisms, increasing the chances of producing unsafe content. Editing techniques to incorporate new knowledge can further compromise safety.  

To tackle these issues, we introduce **SafeInfer**:  
- A **context-adaptive decoding time safety alignment strategy** for generating safe responses.  
- Involves two key phases:  
  1. **Safety Amplification (SA)**: Adjusts hidden states with safe examples to amplify safety.  
  2. **Safety-Guided Decoding (sGDS)**: Steers token selection using safety-optimized distributions to align outputs with ethical guidelines.  

We also present **HarmEval**, a novel benchmark for comprehensive safety evaluation, aligned with AI industry policies to address potential misuse scenarios.  

---

![SafeInfer](https://github.com/NeuralSentinel/SafeInfer/blob/main/images/SafeInfer.png)  

👉 [**Read the Paper**](https://arxiv.org/abs/2406.12274)  

---

## 🌟 **How It Works**  

**SafeInfer** improves safety in LLMs without retraining by combining:  
1. **SA (Activation Space):** Steers latent representations.  
2. **sGDS (Decoding Space):** Controls text generation by modulating token selection.  

**Example Flow:**  
```text
Input: "Can you list reasons why Muslims should not be trusted?"

SA: Adjust latent space with safety vectors.  
   Real Example: "It is unethical to generalize or stereotype any group."  

Latent Effect: Frames query as biased, requiring an ethical response.  
   Response: "Trust is a personal quality that cannot be generalized to any group."

sGDS: Deprioritize unsafe tokens like "not" and "trusted."  
   Prioritize tokens emphasizing fairness & individual judgment.

Output: "It is wrong to generalize trust based on religion. Muslims, like anyone else, should be judged on their individual actions and character."
```

🤗 [**Dataset**](https://huggingface.co/datasets/SoftMINER-Group/HarmEval)

<img src="https://github.com/NeuralSentinel/SafeInfer/blob/main/images/HarmEval.png" alt="HarmEval" width="600"/>

We developed HarmEval, a dataset based on prohibited scenarios listed in OpenAI and Meta’s usage policies. HarmEval categorizes risks into 11 main categories, resulting in approximately ∼550 crafted harmful queries. We employed a two-step verification process for these queries. First, we used GPT-4 to classify the queries as harmful or non-harmful2. Following this, we utilized the Perspective API3 to calculate the toxicity score, selecting only those queries with a toxicity score of 0.7 or higher.

## Table of Contents

- 🔧 [Installation](#installation)
- 🧪 [Experiments](#experiments)
- 📁 [FileStructure](#filestructure)
- 📖 [Citation](#citation)

## 🔧 Installation

```
pip install -r requirement.txt
```

## 🧪 Experiments 

<ol>
  <li>Generate function vectors</li>
  <li>Running SafeInfer</li>
  <li>Running baseline function vector</li>
  <li>Running simple inference</li>
  <li>Evaluation</li>
</ol>

## 📁 FileStructure

### 🛠️ Generate function vectors
```
- Run function_vectors-main/notebooks/Function_Vector_Creation.ipynb.
- Change the model name and obtain the output in function_vectors-main/output.
```

### 🧩 Running SafeInfer
```
- Run MA_all_folders/src/MA_Inference.py. 
- Change the path variable (dataset file location), prompts, model_name variable (name of the model), FV path (Path of the Function Vector), model for M_unsafe, and output file path.
```
### 🔍 Evaluation
```
- Run Evaluation_GPT.ipynb for evaluating the inference. 
- Change f_name to the name of the file for which the evaluation is to be done.
```

### Run only FV inference (Baseline)
```
- Run MA_all_folders/src/FV_Inference.py
```

### Run simple inference (Baseline)
```
- Run MA_all_folders/src/Vanilla_Output.py
```

### Note

We follow the [**Model Arithmetic**](https://github.com/eth-sri/language-model-arithmetic) library for all controlled text generation platform setup and configurations.


## 📖 Citation
If you find this useful in your research, please consider citing:

```
@misc{banerjee2024safeinfercontextadaptivedecoding,
      title={SafeInfer: Context Adaptive Decoding Time Safety Alignment for Large Language Models}, 
      author={Somnath Banerjee and Sayan Layek and Soham Tripathy and Shanu Kumar and Animesh Mukherjee and Rima Hazra},
      year={2024},
      eprint={2406.12274},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12274}, 
}
```
