# %%
import os

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

prompts_QS_UE[1]

max_memory ={0:"46GB", 1:"46GB"}
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
model_name = "meta-llama/Llama-2-7b-chat-hf"
max_memory = {0:"46GB"}#, 1:"46GB"}#,1:"46GB", 2:"0GB"}
EDIT_LAYER = [i for i in range(8,9)]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
model = AutoModelForCausalLM.from_pretrained(model_name,device_map = "auto", max_memory=max_memory)

# %%
#Function vector
FV = torch.tensor(np.load('../../function_vectors-main/outputs/FV_vector_top_head_10_llama_7b.npy'))

import FVPlug as FVP


# %%
#Function vector
model_wrapper = FVP.model_with_FunctionVector(model)
model_fv = model_wrapper.get_model(FV.cuda(), EDIT_LAYER)

#Function vector
# model_fv.model.layers[8]

# %%
def clean_output(output, inst):
    import re
    indexes = output.find(inst)
    start = indexes+len(inst)
    return output[start:]

from tqdm import tqdm
prompts_QS_UE_sliced = [prompts_QS_UE[i:i + 20] for i in range(0, len(prompts_QS_UE), 20)]
Final_Output_safe_edited = []
for x in tqdm(prompts_QS_UE_sliced):
    tokenized_input = tokenizer(x, return_tensors='pt', padding=True, max_length=256)
    generation_output = model_fv.generate(
                        input_ids= tokenized_input['input_ids'].cuda(), #.unsqueeze(0)
                        attention_mask= tokenized_input['attention_mask'].cuda(),#.unsqueeze(0)
                        max_new_tokens=250,
                        do_sample=True,
                        top_k=10,
                        temperature = 0.45,
                        num_return_sequences=1,
                        #eos_token_id=[104,193,tokenizer.eos_token_id]
                    )
    Fout =  [tokenizer.decode(x_out,skip_special_tokens=True) for x_out in generation_output.detach().cpu().numpy().tolist()]
    Final_Output_safe_edited.extend(Fout)

for out in Final_Output_safe_edited:
    print(out)


df_pre = pd.DataFrame()
df_pre['Question'] = ue_qs
df_pre['Safe Answer'] = Final_Output_safe_edited
df_pre.to_csv("../output/LLAMA_EDITED_FV_Deng.csv",index=False)

