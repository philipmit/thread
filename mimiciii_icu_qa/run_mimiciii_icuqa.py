

import io
import time
import re
import gc
import os
import openai
import yaml
import sys
import numpy as np
import pandas as pd
import json




######################################################################################################################################################################### 
#########################################################################################################################################################################
######### MODEL SETUP 


# model_name="meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "codellama/CodeLlama-7b-hf"
# model_name="gpt-4-0613"
model_name="gpt-3.5-turbo-instruct"



# API_KEY = ...
# HF_auth_token = ...
# cache_dir=...


if 'gpt' not in model_name.lower():
    import torch
    import transformers
    from transformers import StoppingCriteria, StoppingCriteriaList
    if 'llama-2' in model_name.lower():
        # llama-2
        from transformers import  LlamaTokenizer
        from transformers import  LlamaForCausalLM
        class CustomStoppingCriteria3(StoppingCriteria):
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.input_length = input_length  # Length of the initial input context
                self.stop_tokens_ids =[1149, 4261]
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                generated_part = input_ids[0, self.input_length:]
                for stop_token_id in [1149, 4261]:
                    if stop_token_id in generated_part:
                        return True
                if generated_part[-1] in [11794]:
                    if len(generated_part)>1:
                        if generated_part[-2] in [29937, 396]: 
                                return True
                return False
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=HF_auth_token)
        model = LlamaForCausalLM.from_pretrained(model_name, device_map = 'auto',cache_dir=cache_dir, torch_dtype=torch.float16, use_auth_token=HF_auth_token)
    elif 'codellama' in model_name.lower():
        # codellama
        from transformers import  CodeLlamaTokenizer
        from transformers import  AutoModelForCausalLM
        # #END#
        class CustomStoppingCriteria3(StoppingCriteria):
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.input_length = input_length  # Length of the initial input context
                self.stop_tokens_ids =[1149, 4261]
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                generated_part = input_ids[0, self.input_length:]
                for stop_token_id in [1149, 4261]:
                    if stop_token_id in generated_part:
                        return True
                if generated_part[-1] in [11794]:
                    if len(generated_part)>1:
                        if generated_part[-2] in [29937, 396]: 
                                return True
                return False
        tokenizer = CodeLlamaTokenizer.from_pretrained(model_name, use_auth_token=HF_auth_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, device_map = 'auto',cache_dir=cache_dir, torch_dtype=torch.float16, use_auth_token=HF_auth_token)
    elif 'llama-3' in model_name.lower():
        # llama-3
        from transformers import  AutoTokenizer
        from transformers import  AutoModelForCausalLM
        class CustomStoppingCriteria3(StoppingCriteria):
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.input_length = input_length  # Length of the initial input context
                self.stop_tokens_ids =[1149, 4261]
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                generated_part = input_ids[0, self.input_length:]
                for stop_token_id in [591, 2228]:
                    if stop_token_id in generated_part:
                        return True
                if generated_part[-1] in [4794]:
                    if len(generated_part)>1:
                        if generated_part[-2] in [2, 674]: 
                                return True
                return False
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_auth_token)
        model=AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto',cache_dir=cache_dir, torch_dtype=torch.float16, use_auth_token=HF_auth_token)
    model.eval()
    def llm(prompt, stop=["\n", "\n"], max_new_tokens_=1000, temp=0):
        with torch.no_grad():
            global input_length
            global generated_text
            try:
                prev_context=prompt0_examples
                input_ids_prev_context=input_ids_prompt0
                past_key_values_prev_context=past_key_values_prompt0
                new_context = '\n\n\n\n\n\nPrompt:\n'+ prompt.split('\n\n\n\n\n\nPrompt:\n')[-1]
                print('catching up...')
                input_ids_new_context = tokenizer.encode('<s>'+new_context, return_tensors="pt")[:,2:]
                if torch.cuda.is_available():
                    input_ids_new_context=input_ids_new_context.to('cuda')
                    input_ids_prev_context=input_ids_prev_context.to('cuda')
                else:
                    input_ids_new_context = input_ids_new_context.to('cpu')
                    input_ids_prev_context=input_ids_prev_context.to('cpu')
                output_combined = model(input_ids=input_ids_new_context, past_key_values=past_key_values_prev_context, use_cache=True)
                next_token_combined = torch.argmax(output_combined.logits[:, -1, :], dim=-1)
                # 
                combined_input_ids = torch.cat([input_ids_prev_context, input_ids_new_context], dim=1)
                # 
                if torch.cuda.is_available():
                    combined_input_ids=combined_input_ids.to('cuda')
                    next_token_combined=next_token_combined.to('cuda')
                else:
                    combined_input_ids=combined_input_ids.to('cpu')
                    next_token_combined=next_token_combined.to('cpu')
                # 
                generated_combined = torch.cat([combined_input_ids, next_token_combined.unsqueeze(-1)], dim=-1)
                generated_combined_attention_mask=torch.ones_like(generated_combined)
                len(generated_combined[0])
                if torch.cuda.is_available():
                    generated_combined=generated_combined.to('cuda')
                    generated_combined_attention_mask=generated_combined_attention_mask.to('cuda')
                else:
                    generated_combined=generated_combined.to('cpu')
                    generated_combined_attention_mask=generated_combined_attention_mask.to('cpu')
                # 
                print('generating...')
                input_length=len(generated_combined[0])
                past_key_values=output_combined.past_key_values
                custom_stopping_criteria = CustomStoppingCriteria3(tokenizer)
                output_final = model.generate(input_ids=generated_combined,attention_mask=generated_combined_attention_mask, stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]),use_cache=True,past_key_values=past_key_values, do_sample=False, num_beams=1, max_new_tokens=1000)
                generated_text=tokenizer.batch_decode([output_final[0]], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                # 
                generate_ids=output_final
                input_length_minus_first_token_from_catch_up_step=input_length-1
                gen_tok=tokenizer.batch_decode([generate_ids[0][input_length_minus_first_token_from_catch_up_step:]], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                if stop[0] in gen_tok:
                    gen_tok=gen_tok[:gen_tok.index(stop_tokens_[0])]
                if stop[1]in gen_tok:
                    gen_tok=gen_tok[:gen_tok.index(stop_tokens_[1])]
                gen_tok_code = None
            except Exception as e:
                print(e)
        return gen_tok
    # 
else:
    openai.api_key = API_KEY
    if model_name=='gpt-3.5-turbo-instruct':
        def llm(prompt, stop=["\n"], max_new_tokens_=1000, temp=0):
            # time.sleep(1)
            global model_name
            global completion_error
            global completion_response
            # max_new_tokens_=500
            # max_new_tokens_=1000
            continue_trying=True
            while(continue_trying):
                try:
                    print('max_new_tokens_: '+str(max_new_tokens_))
                    if temp==0:
                        completion_response = openai.Completion.create(
                        model=model_name,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=max_new_tokens_,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop, 
                        request_timeout=20
                        )
                        continue_trying=False
                    else:
                        completion_response = openai.Completion.create(
                        model=model_name,
                        prompt=prompt,
                        temperature=temp,
                        max_tokens=max_new_tokens_,
                        stop=stop, 
                        request_timeout=20
                        )
                        continue_trying=False
                except openai.error.OpenAIError as e:
                    completion_error=e
                    print(completion_error)
                    if 'Rate limit reached' in str(completion_error):
                        # in case time is redefined elsewhere
                        import time
                        time.sleep(10)
                    elif 'maximum context length' in str(completion_error):
                        if max_new_tokens_>500:
                            max_new_tokens_=max_new_tokens_-150
                        elif max_new_tokens_<50:
                            max_new_tokens_=max_new_tokens_-5
                            if max_new_tokens_ < 5:
                                continue_trying=False
                        else:
                            max_new_tokens_=max_new_tokens_-50
                    elif 'timeout' in str(completion_error):
                        continue_trying = True
                    else:
                        continue_trying=False
            return completion_response["choices"][0]["text"]
    else:
        def llm(prompt, stop=["\n"], max_new_tokens_=1000, temp=0):
            global model_name
            global completion_error
            continue_trying=True
            messages_= [
                {"role": "user", "content": prompt}
                ]  
            while(continue_trying):
                try:
                    print('max_new_tokens_: '+str(max_new_tokens_))
                    completion = openai.ChatCompletion.create(
                    model=model_name,
                    temperature=temp,
                    stop=stop,
                    messages=messages_,
                    request_timeout=60)
                    continue_trying=False
                except openai.error.OpenAIError as e:
                    completion_error=e
                    print(completion_error)
                    if 'Rate limit reached' in str(completion_error):
                        # in case time is redefined elsewhere
                        import time
                        time.sleep(15)
                    elif 'maximum context length' in str(completion_error):
                        if max_new_tokens_>500:
                            max_new_tokens_=max_new_tokens_-150
                        elif max_new_tokens_<50:
                            max_new_tokens_=max_new_tokens_-5
                            if max_new_tokens_ < 5:
                                continue_trying=False
                        else:
                            max_new_tokens_=max_new_tokens_-50
                    elif 'timeout' in str(completion_error):
                        continue_trying = True
                    else:
                        continue_trying=False
            return completion.choices[0].message.content



#########################################################################################################################################################################
#########################################################################################################################################################################




##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
######### BENCHMARK SETUP 

import requests
import ast
from scipy import stats
from scipy.stats import pearsonr

benchmark = 'mimic'

DATA_DIR = 'data_table.txt'

stop_tokens_=['=>','#EN']
max_new_tokens_task = 1000
start_ = '#START#'
close_token='<='

question_df = pd.read_csv('./mimiciii_icu_qa_benchmark.txt',sep='\t')
question_df
question_list_all  = question_df['question'].tolist()
question_df.shape

test_set_size=len(question_list_all)
t_max = 200

prompt0_examples="""Prompt:
MIMIC QUESTION: Was the average systolic blood pressure of patient X from hour 10 to 20 of their ICU stay higher or lower than the average among all patients who expired in the hospital?


Response:
#START#
Here is a table showing the systolic blood pressure of patient X from hour 10 to 20 of their ICU stay: =>{table_dir}<=

Based on the data in {table_dir} the average systolic blood pressure of patient X was =>138.18<= 

The average systolic blood pressure among all patients who expired in the hospital was =>120.81<=

138.18 is higher than 120.81. Therefore, the average systolic blood pressure of patient X from hour 10 to 20 was higher.

Answer choices from question: "higher or lower"
Final Answer: higher
#END#






Prompt:
MIMIC QUESTION: For patient X, was the maximum temperature higher or lower compared to patient Z?


Response:
#START#
Here is a table showing the temperature of patient X and patient Z during their ICU stay: =>{table_dir}<=

Based on the data in {table_dir} the maximum temperature of patient X was =>95.18<= and for patient Z was =>101.43<= 

95.18 is lower than 101.43. Therefore, the maximum temperature of patient X was lower.

Answer choices from question: "higher or lower"
Final Answer: lower
#END#






Prompt:
Here is a table showing the heart rate of patient X from hour 0 to 48 of their ICU stay: =>


Response:
#START#

timeframe = 'hour 0 to 48'
time1 = 0
time2 = 48
patient1_ID = 'X'
variable = 'heart rate'

data_patient1=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1_ID)+"_episode1_timeseries.csv")

> print(data_patient1.columns) =>
# Index(['Hours', 'Capillary refill rate', 'Diastolic blood pressure',
#        'Fraction inspired oxygen', 'Glascow coma scale eye opening',
#        'Glascow coma scale motor response', 'Glascow coma scale total',
#        'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
#        'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
#        'Systolic blood pressure', 'Temperature', 'Weight', 'pH'],
#       dtype='object')
# <=

variable_col_name = 'Heart Rate'
data_patient1 = data_patient1[['Hours', variable_col_name]]
data_patient1 = data_patient1.dropna(subset=[variable_col_name])
table = data_patient1[(data_patient1['Hours']>=time1) & (data_patient1['Hours']<=time2)]
table_dir = DATA_DIR
table.to_csv(table_dir, sep='\t', index=False) 
print('{table_dir}')
#END#






Prompt:
MIMIC QUESTION: For patient X, at what timepoint did heart rate decrease the most?


Response:
#START#
Here is a table showing the heart rate of patient X from hour 0 to 48 of their ICU stay: =>{table_dir}<=

Based on the data in {table_dir} the hour of the biggest decrease in heart rate of patient X was =>22<= 

Final Answer: 22
#END#






Prompt:
Here is a table showing the temperature of patient X and patient Z during their ICU stay: =>


Response:
#START#

timeframe = None
patient1_ID = 'X'
patient2_ID = 'Z'
variable = 'temperature'

# The {variable} for patient number 1, {patient1_ID}, was =>{data_patient1}<= 
# The {variable} for patient number 2, {patient2_ID}, was =>{data_patient2}<= 

table = pd.concat([data_patient1, data_patient2], axis=1)
table.columns = ['Hours', variable + ' for patient ' + str(patient1_ID), 'Hours', variable + ' for patient ' + str(patient2_ID)]
table_dir = DATA_DIR
table.to_csv(table_dir, sep='\t', index=False) 
print('{table_dir}')
#END#





Prompt:
MIMIC QUESTION: Is glucose for patient X expected to be above or below the lowest average glucose among patients with diabetes at hour 27 based on the data from hour 11 to 19?


Response:
#START#
Here is a table showing the glucose of patient X from hour 11 to 19 of their ICU stay: =>{table_dir}<=

Based on the data in {table_dir} the glucose of patient X at hour 27 is expected to be =>51.5<= 

The lowest average glucose among patients with diabetes was =>58.1<=

51.5 is lower than 58.1. Therefore, the glucose of patient X at hour 27 is expected to be lower.

Answer choices from question: "above or below"
Final Answer: below
#END#






Prompt:
# The temperature for patient number 2, Z, was =>


Response:
#START#

patient2_ID = 'Z'
variable = 'temperature'

data_patient2=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient2_ID)+"_episode1_timeseries.csv")

> print(data_patient2.columns) =>
# Index(['Hours', 'Capillary refill rate', 'Diastolic blood pressure',
#        'Fraction inspired oxygen', 'Glascow coma scale eye opening',
#        'Glascow coma scale motor response', 'Glascow coma scale total',
#        'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
#        'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
#        'Systolic blood pressure', 'Temperature', 'Weight', 'pH'],
#       dtype='object')
# <=

variable_col_name = 'Temperature'
data_patient2 = data_patient2[['Hours', variable_col_name]]
data_patient2 = data_patient2.dropna(subset=[variable_col_name])

print('{data_patient2}') 
#END#






Prompt:
Based on the data in data_table.txt the average systolic blood pressure of patient X was =>


Response:
#START#

patient1_ID = 'X'
table_dir = 'data_table.txt'
variable = 'systolic blood pressure'

table = pd.read_csv(table_dir, sep='\t')

>>> print(table.columns) =>
# Index(['Hours', 'Systolic blood pressure'], dtype='object')
<=

variable_col_name = 'Systolic blood pressure'
y = table[variable_col_name]
y = [x for x in y if not math.isnan(x)]
y = [x for x in y if not None]
y_mean = np.mean(y)
y_mean = round(y_mean, 2)

print(y_mean) 
#END#






Prompt:
Based on the data in data_table.txt the hour of the biggest decrease in heart rate of patient X was =>


Response:
#START#

patient1_ID = 'X'
table_dir = 'data_table.txt'
variable = 'heart rate'

table = pd.read_csv(table_dir, sep='\t')

>>> print(table.columns) =>
# Index(['Hours', 'Heart Rate'], dtype='object')
<=

variable_col_name = 'Heart Rate'
value_diff = np.diff(table[variable_col_name])
hour_diff_idx_min = np.argmin(value_diff)
hour_dif_min = round(table['Hours'][hour_diff_idx_min])

print(hour_dif_min)
#END#






Prompt:
Based on the data in data_table.txt the glucose of patient X at hour 27 is expected to be =>


Response:
#START#

patient1_ID = 'X'
table_dir = 'data_table.txt'
variable = 'glucose'
future_time = 27

table = pd.read_csv(table_dir, sep='\t')

> print(table.columns) =>
# Index(['Hours', 'Glucose'], dtype='object')
# <=

variable_col_name = 'Glucose'
X = table['Hours']
y = table[variable_col_name]
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
expected_value = slope * int(future_time) + intercept
expected_value = round(expected_value, 2)

print(expected_value)
#END#






Prompt:
Based on the data in data_table.txt the maximum temperature of patient X was =>95.18<= and for patient Z was =>


Response:
#START#

patient1_ID = 'X'
patient2_ID = 'Z'
table_dir = 'data_table.txt'
variable = 'temperature'

table = pd.read_csv(table_dir, sep='\t')

>>> print(table.columns) =>
# Index(['Hours', 'temperature for patient X', 'Hours.1', 
#        'temperature for patient Z'],
#       dtype='object')
# <=

variable_col_name = 'temperature for patient ' + str(patient2_ID)
y = table[variable_col_name]
y = [x for x in y if not math.isnan(x)]
y = [x for x in y if not None]
y_max = np.max(y)
y_max = round(y_max, 2)

print(y_max)
#END#






Prompt:
The average systolic blood pressure among all patients who expired in the hospital was =>


Response:
#START#

variable = 'systolic blood pressure'
table = pd.read_csv(MIMIC_DATA_DIR+'/all_stays.csv')

> print(table.columns) =>
# Index(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE',
#        'INTIME', 'OUTTIME', 'LOS', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME',
#        'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'DOB', 'DOD', 'AGE',
#        'MORTALITY_INUNIT', 'MORTALITY', 'MORTALITY_INHOSPITAL'],
#       dtype='object')
# <=

filter_col_name = 'MORTALITY_INHOSPITAL'
target_patient_group_IDs = list(table[table[filter_col_name]==1]['SUBJECT_ID'])
target_patient_group_IDs = np.unique(target_patient_group_IDs)

target_value_patient_list = []
for idx in range(len(target_patient_group_IDs)):
    try:
        data_patient=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(target_patient_group_IDs[idx])+"_episode1_timeseries.csv")
        data_patient = data_patient[['Hours', variable_col_name]]
        data_patient = data_patient.dropna(subset=[variable_col_name])
        if len(data_patient)<=2:
            continue
        target_value_patient_list.append(np.mean(list(data_patient[variable_col_name])))
    except Exception as e:
        continue

target_value_patient_list_mean = np.mean(target_value_patient_list)
target_value_patient_list_mean = round(target_value_patient_list_mean, 2)

print(target_value_patient_list_mean)
#END#






Prompt:
The lowest average glucose among patients with diabetes was =>


Response:
#START#

variable = 'glucose'
pheno = pd.read_csv(MIMIC_DATA_DIR+'/phenotype_labels.csv')
table = pd.read_csv(MIMIC_DATA_DIR+'/all_stays.csv')
table = pd.concat([table[['SUBJECT_ID']], pheno], axis=1)

> print(table.columns) =>
# Index(['SUBJECT_ID', 'Acute and unspecified renal failure',
#        'Acute cerebrovascular disease', 'Acute myocardial infarction',
#        'Cardiac dysrhythmias', 'Chronic kidney disease',
#        'Chronic obstructive pulmonary disease and bronchiectasis',
#        'Complications of surgical procedures or medical care',
#        'Conduction disorders', 'Congestive heart failure; nonhypertensive',
#        'Coronary atherosclerosis and other heart disease',
#        'Diabetes mellitus with complications',
#        'Diabetes mellitus without complication',
#        'Disorders of lipid metabolism', 'Essential hypertension',
#        'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
#        'Hypertension with complications and secondary hypertension',
#        'Other liver diseases', 'Other lower respiratory disease',
#        'Other upper respiratory disease',
#        'Pleurisy; pneumothorax; pulmonary collapse',
#        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
#        'Respiratory failure; insufficiency; arrest (adult)',
#        'Septicemia (except in labor)', 'Shock'],
#       dtype='object')
# <=

filter_col_name = 'Diabetes mellitus with complications'
target_patient_group_IDs = list(table[table[filter_col_name]==1]['SUBJECT_ID'])
target_patient_group_IDs = np.unique(target_patient_group_IDs)

target_value_patient_list = []
for idx in range(len(target_patient_group_IDs)):
    try:
        data_patient=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(target_patient_group_IDs[idx])+"_episode1_timeseries.csv")
        data_patient = data_patient[['Hours', variable_col_name]]
        data_patient = data_patient.dropna(subset=[variable_col_name])
        if len(data_patient)<=2:
            continue
        target_value_patient_list.append(np.mean(list(data_patient[variable_col_name])))
    except Exception as e:
        continue

target_value_patient_list_min = np.min(target_value_patient_list)
target_value_patient_list_min = round(target_value_patient_list_min, 2)

print(target_value_patient_list_mean)
#END#
"""





if not 'gpt' in model_name.lower():
    with torch.no_grad():
        input_ids_prompt0 = tokenizer.encode(prompt0_examples, return_tensors="pt")
        input_ids_prompt0 = input_ids_prompt0.to('cuda')
        output_prompt0 = model(input_ids = input_ids_prompt0, use_cache=True)
        past_key_values_prompt0 = output_prompt0.past_key_values


def check_answer(model_response, mimic_answer):
    try:
        mimic_answer = float(mimic_answer)
        mimic_answer = int(mimic_answer)
        model_answer = model_response.split('\nFinal Answer:')[-1].strip()
        model_answer = float(model_answer)
        model_answer = int(model_answer)
        if model_answer==mimic_answer:
            is_correct='yes'
        else:
            is_correct='no'
    except Exception as e:
        model_answer=None
        is_correct='no'
        try:
            model_answer = model_response.split('\nFinal Answer:')[-1].strip()
            mimic_answer = mimic_answer.strip().lower()
            model_answer = model_answer.strip().lower()
            if model_answer in mimic_answer or mimic_answer in model_answer:
                is_correct='yes'
            else:
                is_correct='no'
        except Exception as e:
            print(e)
    return model_answer, is_correct




##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################


# same thread function used for all benchmarks
def thread(c, Y):
    global t
    t = t+1
    if t < t_max:
        gc.collect()
        c_plus_Y = prompt0_examples + '\n\n\n\n\n\nPrompt:\n'+ c  + '\n\n\nResponse:\n'+start_+'\n' + Y
        Y = Y + llm(c_plus_Y, stop=stop_tokens_, max_new_tokens_=max_new_tokens_task)
        thread_status = get_thread_status(Y)
        if thread_status == 'spawn':
            Y = Y + psi( thread( phi(Y), '') )
            if not done:
                return thread(c, Y)
        elif thread_status == 'action':
            action, observation=get_action_and_observation(Y)
            Y = Y + observation 
            if not done:
                return thread(c, Y) 
        else: 
            return Y



def phi(parent_token_sequ):
    global thread_context
    define_variables(parent_token_sequ)
    thread_context = parent_token_sequ.split('\n')[-1]
    thread_context = thread_context + '=>'
    variable_names = list(np.unique(re.findall(r"\{(.*?)\}", thread_context)))
    if len(variable_names)>0:
        replace_var_exec='thread_context=thread_context.format('
        for variable_name in variable_names:
            replace_var_exec=replace_var_exec+variable_name+'='+variable_name+','
        replace_var_exec=replace_var_exec[:-1]+')'
        try:
            exec(replace_var_exec, globals())
        except Exception as e_replace_var:
            print('')
    return thread_context


def psi(token_sequ):
    if type(token_sequ) is str:
        define_variables(token_sequ)
        if 'print' in token_sequ:
            print_line = '\nprint('+token_sequ.split("print(")[-1].split(")")[0] + ')'
            output_buffer = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = output_buffer
            try:
                exec(print_line, globals())
            except Exception as e:
                print(e)
            sys.stdout = original_stdout
            observation = output_buffer.getvalue()
            output_buffer.close()
            observation = observation.strip()
            if '\n' in observation:
                observation = '\n'+observation+'\n'
            observation = '=>' + observation + close_token
            return observation
        else: 
            return ''
    else:
        return ''





def get_action_and_observation(token_sequ):
    global reward
    global done
    resp_last_line = token_sequ.split('\n')[-1]
    define_variables(token_sequ)
    try:
        action=get_action(resp_last_line)
        # 
        output_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer
        try:
            exec(action, globals())
        except Exception as e:
            print(e)
        sys.stdout = original_stdout
        observation = output_buffer.getvalue()
        output_buffer.close()
        observation = observation.strip()
        # 
        observation = observation.strip()
        observation = observation.replace('\n', '\n# ')
        if '\n' in observation:
            observation = '\n# ' + observation + '\n# '
        observation = '=>' + observation + close_token
        return action, observation
    except Exception as e:
        print(e)
        return None, None



def print_res_df(res_df):
    print('********************************************************************************************************************************************************')
    print('********************************************************************************************************************************************************')
    print('********************************************************************************************************************************************************')
    print('****************************************************************** res_df')
    print(res_df.to_string(index=True))
    # print(res_df.to_string(index=False))
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    import time
    try:
        time.sleep(5)
    except Exception as e:
        print(e)



def define_variables(token_sequ):
    try:
        token_sequ = token_sequ.replace('\n>','\n# >')
        exec(token_sequ, globals())
    except Exception as e_resp:
        print('')

def get_action(resp_last_line_input):
    action=resp_last_line_input.split('> ')[-1].strip()
    return action


def get_thread_status(token_sequ):
    if token_sequ[-1] == close_token or token_sequ[-1] == '\n' or '\nprint(' in token_sequ.split('\n')[-1]:
        return 'end'
    elif (token_sequ.split('\n')[-1].startswith('>') or token_sequ.split('\n')[-1].startswith('# >')) and not token_sequ.split('\n')[-1].strip().endswith('<='):
        return 'action'
    else:
        return 'spawn'






question_list=[]
name_list=[]
is_correct_list=[]
for test_idx in range(test_set_size):
    if benchmark == 'mimic':
        question=question_list_all[test_idx]
        name=str(test_idx)
        question=question.replace(',','')
        question = 'MIMIC QUESTION: '+question
        mimic_answer = question_df['mimic_answer'].tolist()[test_idx]
        context0=question
        # 
        try: 
            os.system('rm ' + DATA_DIR)
        except Exception as e:
            print(e)
    # 
    print(question)
    question_list.append(question)
    name_list.append(name)
    print('\n')
    # 
    t=0
    done=False
    reward=0
    try:
        main_thread_token_sequ = thread(context0, '')
    except Exception as e:
        print(e)
    # 
    model_answer, is_correct = check_answer(main_thread_token_sequ, mimic_answer)
    print(model_answer)
    print(mimic_answer)
    is_correct_list.append(is_correct)
    res_df=pd.DataFrame({'name':name_list,'question':question_list,'is_correct':is_correct_list})
    print_res_df(res_df)







