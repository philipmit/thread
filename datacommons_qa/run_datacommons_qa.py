

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

benchmark = 'dc'

URL_EXPLORE = 'https://datacommons.org/api/explore/detect'
URL_obs = 'https://api.datacommons.org/v2/observation'
HEADERS = {"Content-Type": "application/json"}
JSON = {"contextHistory": [],"dc": ""}
DATA_DIR = 'data_table.txt'

stop_tokens_=['=>','#EN']
max_new_tokens_task = 1000
start_ = '#START#'
close_token='<='

question_df = pd.read_csv('./datacommons_qa_benchmark.txt',sep='\t')
question_df
question_list_all  = question_df['question'].tolist()
question_df.shape

test_set_size=len(question_list_all)
t_max = 200


prompt0_examples="""Prompt:
Question: From 2018 to 2021 was the percentage of people with diabetes increasing faster in Willows or Villa Ridge?


Response:
#START#
Here is a table showing the percentage of people with diabetes during 2018 to 2021 for Willows and Villa Ridge: =>{table_dir}<=.

Based on the data in {table_dir} from 2018 to 2021 the slope of the percentage of people with diabetes in Willows is =>0.3357<= and in Villa Ridge is =>0.1642<=.

Since 0.3357 in Willows is greater than 0.1642 in Villa Ridge, the percentage of people with diabetes is increasing faster in Willows.

Answer choices from question: "Willows or Villa Ridge"
Final Answer: Willows
#END#






Prompt:
Question: Compared to the national average was the percentage of individuals who have received cholesterol screening in Seattle in 2016 higher or lower?


Response:
#START#
Here is a table showing the percentage of individuals who have received cholesterol screening during 2016 for Seattle and United States: =>{table_dir}<=.

Based on the data in {table_dir} in 2016 the percentage of individuals who have received cholesterol screening in Seattle is =>7.9<= and in United States is =>6.1<=. 

Since 7.9 in Seattle is greater than 6.1 in United States, the percentage of individuals who have received cholesterol screening in Seattle is higher.

Answer choices from question: "higher or lower"
Final Answer: higher
#END#






Prompt:
Question: Was the number of people with employer-based health insurance in Belpre increasing or decreasing from 2018 to 2020?


Response:
#START#
Here is a table showing the number of people with employer-based health insurance during 2018 to 2020 for Belpre: =>{table_dir}<=.

Based on the data in {table_dir} from 2018 to 2020 the slope of the number of people with employer-based health insurance in Belpre is =>-31.5<=.

Since -31.5 is negative, the number of people with employer-based health insurance in Belpre is decreasing.

Answer choices from question: "increasing or decreasing"
Final Answer: decreasing
#END#






Prompt:
Question: Was there a positive or negative correlation in the prevalence of health insurance coverage in New Jersey and Virginia based on the data from 2019 to 2021?


Response:
#START#
Here is a table showing the prevalence of health insurance coverage during 2019 to 2021 for New Jersey and Virginia: =>{table_dir}<=.

Based on the data in {table_dir} from 2019 to 2021 the correlation in the prevalence of health insurance coverage in New Jersey and Virginia was =>0.13<=.

Since 0.13 is positive, the correlation in the prevalence of health insurance coverage in New Jersey and Virginia was positive.

Answer choices from question: "positive or negative"
Final Answer: positive
#END#






Prompt:
Question: Using data from 2019 to 2022 in Pacifica California is the rate of individuals aged 40 to 64 who have health insurance coverage expected to be above or below Loyall Kentucky in 2039?


Response:
#START#
Here is a table showing the rate of individuals aged 40 to 64 who have health insurance coverage during 2019 to 2022 for Pacifica California and Loyall Kentucky: =>{table_dir}<=.

Based on the data in {table_dir} from 2019 to 2022 the expected value in 2039 of the rate of individuals aged 40 to 64 who have health insurance coverage in Pacifica California is =>61.3<= and in Loyall Kentucky is =>64.9<=.

Since 61.3 in Pacifica California is less than 64.9 in Loyall Kentucky, the rate of individuals aged 40 to 64 who have health insurance coverage in Pacifica California is expected to be below.

Answer choices from question: "above or below"
Final Answer: below
#END#






Prompt:
Question: From 2014 to 2017 was the minimum value of the prevalence rate of individuals aged 18 to 64 who have health insurance higher in Blevins Arkansas or CORTLAND New York?


Response:
#START#
Here is a table showing the prevalence rate of individuals aged 18 to 64 who have health insurance during 2014 to 2017 for Blevins Arkansas and CORTLAND New York: =>{table_dir}<=.

Based on the data in {table_dir} from 2014 to 2017 the minimum value of the prevalence rate of individuals aged 18 to 64 who have health insurance in Blevins Arkansas is =>50.4<= and in CORTLAND New York is =>57.8<=.

Since 50.4 in Blevins Arkansas is less than 57.8 in CORTLAND New York, the minimum value of the prevalence rate of individuals aged 18 to 64 who have health insurance is higher in CORTLAND New York.

Answer choices from question: "Blevins Arkansas or CORTLAND New York"
Final Answer: CORTLAND New York 
#END#






Prompt:
Here is a table showing the percentage of people with diabetes during 2018 to 2021 for Willows and Villa Ridge: =>


Response:
#START#
location1 = 'Willows'
location2 = 'Villa Ridge'
timeframe = '2018 to 2021'
variable = 'percentage of people with diabetes'

# I need to create a table with {variable} during {timeframe} for {location1} and {location2}.

# First, I need to retreive the {variable} during {timeframe} for location number 1: {location1} =>You have retrieved {data_location1_in_timeframe}<=.

# Next, I need to retreive the {variable} during {timeframe} for location number 2: {location2} =>You have retrieved {data_location2_in_timeframe}<=.

# Finally, I need to create a table with the retrieved data.

data_location1_in_timeframe = [data_location1_in_timeframe] if not isinstance(data_location1_in_timeframe, list) else data_location1_in_timeframe
data_location2_in_timeframe = [data_location2_in_timeframe] if not isinstance(data_location2_in_timeframe, list) else data_location2_in_timeframe

timeframe_dates = np.unique([x['date'] for x in data_location1_in_timeframe] + [x['date'] for x in data_location2_in_timeframe])
data_location1_in_timeframe_values = []
data_location2_in_timeframe_values = []
for time in timeframe_dates:
    if time not in [x['date'] for x in data_location1_in_timeframe]:
        data_location1_in_timeframe_values.append(None)
    else:
        data_location1_in_timeframe_values.append([x['value'] for x in data_location1_in_timeframe if x['date']==time][0])
    if time not in [x['date'] for x in data_location1_in_timeframe]:
        data_location2_in_timeframe_values.append(None)
    else:
        data_location2_in_timeframe_values.append([x['value'] for x in data_location2_in_timeframe if x['date']==time][0])

table = pd.DataFrame({'date': timeframe_dates, location1: data_location1_in_timeframe_values, location2: data_location2_in_timeframe_values})
table_dir = DATA_DIR
table.to_csv(table_dir, sep='\t', index=False)

print('{table_dir}')
#END#






Prompt:
I need to retreive the percentage of people with diabetes during 2018 to 2021 for location number 2: Villa Ridge =>


Response:
#START#
location2 = 'Villa Ridge'
timeframe = '2018 to 2021'
time1 = '2018'
time2 = '2021'
variable = 'percentage of people with diabetes'

# I need to retreive the {variable} during {timeframe} for location 2: {location2}.

# I will first get the variable_ID and location2_ID.

query = f'{variable} in {location2}'
response = requests.post(URL_EXPLORE + '?q=' + query, headers=HEADERS, json=JSON)
response = json.loads(response.text)

> print(response['variables']) =>['dc/topic/Diabetes', 'Percent_Person_WithDiabetes', 'Percent_Person_20OrMoreYears_WithDiabetes', 'dc/topic/PopulationWithDiabetesByAge', 'Count_Person_20To79Years_Diabetes_AsFractionOf_Count_Person_20To79Years', 'WHO/SDG_SH_DTH_RNCOM_DiabetesMellitus', 'dc/topic/DiabetesFemalePopulationByAge', 'dc/nh3s4skee5483']<=

variable_ID = 'Percent_Person_WithDiabetes'

> print(response['entities']) =>['geoId/2976192']<=

location2_ID = 'geoId/2976192'

# I will now use the location2_ID and variable_ID to get the data.

data_location2_all = collect_data_commons(location2_ID, variable_ID)
data_location2_in_timeframe = [x for x in data_location2_all if (int(x['date'])>=int(time1) and int(x['date'])<=int(time2))]

> print(data_location2_in_timeframe) =>[{'date': '2018', 'value': 10.1}, {'date': '2020', 'value': 10.8}, {'date': '2021', 'value': 11.1}]<=

print('You have retrieved {data_location2_in_timeframe}')
#END#






Prompt:
Based on the data in data_table.txt from 2018 to 2021 the slope of the percentage of people with diabetes in Willows is =>0.33571428571428585<= and in Villa Ridge is =>


Response:
#START#
table_dir = 'data_table.txt'
variable = 'percentage of people with diabetes'
location1 = 'Willows'
location2 = 'Villa Ridge'

# I need to find the slope of the {variable} in {location2} using the data in {table_dir}.

table = pd.read_csv(table_dir, sep='\t')

> print(table) =>
#    date  Willows  Villa Ridge
# 0  2018     10.1          8.2
# 1  2020     10.8          8.5
# 2  2021     11.1          8.7
# <=

location2_col_name = 'Villa Ridge'

X = table['date']
y = table[location2_col_name]
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

print(slope)
#END#






Prompt:
Based on the data in data_table.txt from 2019 to 2022 the expected value in 2039 of the rate of individuals aged 40 to 64 who have health insurance coverage in Pacifica California is =>61.3<= and in Loyall Kentucky is =>

Response:
#START#
table_dir = 'data_table.txt'
variable = 'rate of individuals aged 40 to 64 who have health insurance coverage'
location1 = 'Pacifica California'
location2 = 'Loyall Kentucky'
future_time = '2039'

# I need to find the expected value in {future_time} the {variable} in {location2} using the data in {table_dir}.

table = pd.read_csv(table_dir, sep='\t')

> print(table) =>
#    date  Pacifica California  Loyall Kentucky
# 0  2019                 48.1             51.2
# 1  2020                 51.8             53.5
# 2  2021                 52.1             53.7
# <=

location2_col_name = 'Villa Ridge'

X = table['date']
y = table[location2_col_name]
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
expected_value = slope * int(future_time) + intercept

print(expected_value)
#END#






Prompt:
Based on the data in data_table.txt from 2019 to 2021 the correlation in the prevalence of health insurance coverage in New Jersey and Virginia was =>

Response:
#START#
table_dir = 'data_table.txt'
variable = 'percentage of people with diabetes'
location1 = 'New Jersey'
location2 = 'Virginia'

# I need to find the correlation of the {variable} in {location1} and {location2} using the data in {table_dir}.

table = pd.read_csv(table_dir, sep='\t')

> print(table) =>
#    date  New Jersey      Virginia
# 0  2019        60.1          58.2
# 1  2020        62.8          58.5
# 2  2021        63.1          58.7
# <=

location1_col_name = 'New Jersey'
location2_col_name = 'Virginia'

y1 = table[location1_col_name]
y2 = table[location2_col_name]
corr, _ = pearsonr(y1, y2)

print(corr)
#END#






Prompt:
Based on the data in data_table.txt from 2014 to 2017 the minimum value of the prevalence rate of individuals aged 18 to 64 who have health insurance in Blevins Arkansas is =>50.4<= and in CORTLAND New York is =>


Response:
#START#
table_dir = 'data_table.txt'
variable = 'percentage of people with diabetes'
location1 = 'Blevins Arkansas'
location2 = 'CORTLAND New York'

# I need to find the value of the {variable} in {location2} using the data in {table_dir}.

table = pd.read_csv(table_dir, sep='\t')

> print(table) =>
#    date   Blevins Arkansas   CORTLAND New York
# 0  2018               50.4                59.9
# 1  2019               54.9                57.8
# 2  2020               55.6                59.5
# <=

location2_col_name = 'CORTLAND New York'

y = table[location2_col_name]
y_min = np.min(y)

print(y_min)
#END#
"""





if not 'gpt' in model_name.lower():
    with torch.no_grad():
        input_ids_prompt0 = tokenizer.encode(prompt0_examples, return_tensors="pt")
        input_ids_prompt0 = input_ids_prompt0.to('cuda')
        output_prompt0 = model(input_ids = input_ids_prompt0, use_cache=True)
        past_key_values_prompt0 = output_prompt0.past_key_values


def check_answer(model_response, dc_answer):
    model_answer=None
    is_correct='no'
    try:
        model_answer = model_response.split('\nFinal Answer:')[-1].strip()
        dc_answer = dc_answer.strip().lower()
        model_answer = model_answer.strip().lower()
        if model_answer in dc_answer or dc_answer in model_answer:
            is_correct='yes'
        else:
            is_correct='no'
    except Exception as e:
        print(e)
    return model_answer, is_correct



def collect_data_commons1(entity_ID, variable_ID):
    url = "https://api.datacommons.org/v2/observation"
    params = {
    "key": "AIzaSyCTI4Xz-UW_G2Q2RfknhcfdAnTHq5X5XuI",
    "entity.dcids": entity_ID, 
    "select": ["entity", "variable", "value", "date"],
    "variable.dcids": variable_ID 
    }
    response = requests.get(url, params=params)
    obs_data = json.loads(response.text)
    entity_data = obs_data["byVariable"][variable_ID]["byEntity"][entity_ID]
    if "orderedFacets" not in list(entity_data.keys()):
        return None
    seq_data = entity_data["orderedFacets"][0]['observations']
    return seq_data






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
    comma_idx = thread_context.find(', ')
    if comma_idx > -1:
        thread_context = thread_context[comma_idx+2:]
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
    if benchmark == 'dc':
        question=question_list_all[test_idx]
        name=str(test_idx)
        question=question.replace(',','')
        question = 'Question: '+question
        dc_answer = question_df['dc_answer'].tolist()[test_idx]
        context0=question
        # 
        # reset variables used in prompt0_examples
        location1=None
        location2=None
        timeframe_dates=None
        data_location1_in_timeframe=None
        data_location2_in_timeframe=None
        data_location1_in_timeframe_values=None
        data_location2_in_timeframe_values=None
        X=None
        y=None
        slope=None
        expected_value=None
        corr=None
        table=None
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
    model_answer, is_correct = check_answer(main_thread_token_sequ, dc_answer)
    print(model_answer)
    print(dc_answer)
    is_correct_list.append(is_correct)
    res_df=pd.DataFrame({'name':name_list,'question':question_list,'is_correct':is_correct_list})
    print_res_df(res_df)







