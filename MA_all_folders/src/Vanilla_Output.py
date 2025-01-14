# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"#,4"
import pandas as pd
path = "../Dataset/DengerousQA/Dangerous_QA_200.csv"

df = pd.read_csv(path)
ue_qs = df['Question'].tolist()

prompts_QS_UE = [] 
for ue_q in ue_qs:
    q = ue_q.strip(" Only provide answer of the given question and do not print anything else. ")
    # temp_prompt = f'''Task: Write a function that performs {q}. Provide step-by-step code generation.
    # 1. Define the function signature.
    # 2. Check if input parameters are valid.
    # 3. Initialize any necessary variables.
    # 4. Implement the main logic of the function.
    # 5. Test the function with sample inputs.'''
    q = " Only provide answer of the given question and do not print anything else. "+q
    # prompts_QS_UE.append(temp_prompt)
    prompts_QS_UE.append(q)
     # Provide step-by-step code generation.

max_memory ={0:"46GB", 1:"46GB"}
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", max_memory=max_memory, device_map="auto")#,cache_dir = "/mnt/data/rima")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", max_memory=max_memory, device_map="auto")#,cache_dir = "/mnt/data/rima")

def clean_output(output):
    inst = " Only provide answer of the given question and do not print anything else. "
    import re
    indexes = output.find(inst)
    start = indexes+len(inst)
    #print(start)
    return output[start:]

# %%
from tqdm import tqdm
output_pre = []

prompts_QS_UE_sliced = [prompts_QS_UE[i:i + 50] for i in range(0, len(prompts_QS_UE), 50)]
print(len(prompts_QS_UE_sliced))

for prompt_text_slice in tqdm(prompts_QS_UE_sliced):
    batch = tokenizer(prompt_text_slice, return_tensors='pt', padding=True, max_length= 512)

    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'), 
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=200 #512
        )
    pre_edit_output = [tokenizer.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
    output_pre.extend(pre_edit_output)
    break

for i in range(len(output_pre)):
    output_pre[i] = output_pre[i].replace(prompts_QS_UE[i],'')
    print(output_pre[i])

df_pre = pd.DataFrame({"question": ue_qs, 'prompt': prompts_QS_UE, 'Output_Pre': output_pre})
df_pre.to_csv("../output/MISTRAL_PRE_MA_MAFV_OURDATASETQA_Clus_"+str(c_i)+".csv",index=False)
