# %%
import os   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
path = "../Dataset/DengerousQA/Dangerous_QA_200.csv"
df = pd.read_csv(path)
df_ue = df

ue_qs = df_ue['Question'].tolist()
prompts_QS_UE = [] 
for ue_q in ue_qs:
    q = ue_q.strip(" Only provide answer of the given question and do not print anything else. ")
    q = " Only provide answer of the given question and do not print anything else. "+q
    # temp_prompt = f'''Task: Write a function that performs {q}.
    # 1. Define the function signature.
    # 2. Check if input parameters are valid.
    # 3. Initialize any necessary variables.
    # 4. Implement the main logic of the function.
    # 5. Test the function with sample inputs.'''
    # prompts_QS_UE.append(temp_prompt)
    
    prompts_QS_UE.append(q)

max_memory ={0:"46GB"}#, 1:"9GB"}
from transformers import LlamaTokenizer, AutoTokenizer
from transformers import LlamaForCausalLM, AutoModelForCausalLM


from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import numpy as np
import torch
model_name = "meta-llama/Llama-2-7b-chat-hf"
max_memory = {0:"46GB",1:"46GB"}#,1:"46GB", 2:"0GB"}
EDIT_LAYER = [i for i in range(8,9)]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
model = AutoModelForCausalLM.from_pretrained(model_name,device_map = "auto", max_memory=max_memory)

#Function vector
FV = torch.tensor(np.load('../../function_vectors-main/outputs/FV_vector_top_head_10_llama_7b.npy'))

#Function vector
import FVPlug as FVP

model_wrapper = FVP.model_with_FunctionVector(model)
model_fv = model_wrapper.get_model(FV.cuda(), EDIT_LAYER)

#Function vector
# model_fv.model.layers[8]

from model_arithmetic import ModelArithmetic, PromptedLLM, Union
# define model prompt template
prompt_template = lambda formula_string, input_string: f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"

M_unsafe = PromptedLLM("", prompt_template=prompt_template,
                    model = "../../../../sft_fft_llama2_7b_our_SFT_data_0905")

M_inst = PromptedLLM("", prompt_template=prompt_template,
                     model=model_fv)#"meta-llama/Llama-2-7b-chat-hf")

# model arithmetic expression
formula1 = M_inst - 0.99 * Union(M_unsafe, M_inst)

# generate text as usual
ma0 = ModelArithmetic(formula1) #, default_model="meta-llama/Llama-2-7b-chat-hf")


from tqdm import tqdm 
model_arith_out = []

prompts_QS_UE_sliced = [prompts_QS_UE[i:i + 10] for i in range(0, len(prompts_QS_UE), 10)]
for prompt_text_sliced in tqdm(prompts_QS_UE_sliced):   
    output_slice = ma0.generate_text(prompt_text_sliced, max_length = 150, skip_special_tokens=True)
    model_arith_out.extend(output_slice)


# %%
for m in model_arith_out:
    print(m, "\n++++++++++++++++++\n")

# ue_qs = ue_qs[:len(model_arith_out)]
df1= pd.DataFrame()
df1['Question'] = ue_qs
df1['MA-FV Output'] = model_arith_out
df1.to_csv("../output/LLAMA_ALPHA_150_DENGEROUSQA.csv",index=False)

