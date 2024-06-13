

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
from bs4 import BeautifulSoup
from bs4.element import Comment

benchmark = 'webshop'
# WEBSHOP_URL=...

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}


def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    if False:
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class webshopEnv:
    def __init__(self):
        self.sessions = {}
    def step(self, session, action):
        done = False
        observation_ = None
        clicked_p=False
        if action == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}
        elif action == 'return to results':
            self.sessions[session] = self.sessions[session]
        elif action.startswith('think['):
            observation = 'OK.'
        elif action.startswith('search['):
            assert self.sessions[session]['page_type'] == 'init'
            query = action[7:-1]
            self.sessions[session] = {'session': session, 'page_type': 'search',
                                                                'query_string': query, 'page_num': 1}
        elif action.startswith('click['):
            button = action[6:-1]
            if button == 'Buy Now':
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button == 'Next >':
                assert False # ad hoc page limitation
                assert self.sessions[session]['page_type'] == 'search'
                self.sessions[session]['page_num'] += 1
            elif button == '< Prev':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                if self.sessions[session]['page_type'] == 'search':
                    assert False
                    self.sessions[session]['page_num'] -= 1
                elif self.sessions[session]['page_type'] == 'item_sub':
                    self.sessions[session]['page_type'] = 'item'
                elif self.sessions[session]['page_type'] == 'item':
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options'] = {}
                clicked_p=True
            elif button in ACTION_TO_TEMPLATE:
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage'] = button
            else:
                if self.sessions[session]['page_type'] == 'search':
                    assert button in self.sessions[session].get('asins', [])    # must be asins
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                elif self.sessions[session]['page_type'] == 'item':
                    assert 'option_types' in self.sessions[session]
                    assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])    # must be options
                    option_type = self.sessions[session]['option_types'][button]
                    if not 'options' in self.sessions[session]:
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False
        observation, info = webshop_text(**self.sessions[session])
        if observation_:
            observation = observation_
        if clicked_p:
            observation = f'You have clicked {button}.'
        self.sessions[session].update(info)
        reward = info.get('reward', 0.0)
        return observation, reward, done


env = webshopEnv()
env.step('fixed_0','reset')


stop_tokens_=['=>','#EN']
close_token='<='
max_new_tokens_task = 1000
start_ = '#START#'
test_set_size = 100
t_max = 200

# prompt_file='prompt.json'
prompt_d = json.load(open(prompt_file, 'r'))
prompt_d.keys()
prompt0_examples=prompt_d['thread_webshop']

if not 'gpt' in model_name.lower():
    with torch.no_grad():
        input_ids_prompt0 = tokenizer.encode(prompt0_examples, return_tensors="pt")
        input_ids_prompt0 = input_ids_prompt0.to('cuda')
        output_prompt0 = model(input_ids = input_ids_prompt0, use_cache=True)
        past_key_values_prompt0 = output_prompt0.past_key_values




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
    action=get_action(resp_last_line)
    # 
    try:
        res = env.step(idx, action)
        observation = res[0]
        reward = res[1]
        done = res[2]
    except Exception as e:
        if action.split('[')[0] == 'click':
            clicked_item = action.split('[')[1].split(']')[0]
            observation = 'You have clicked ' + clicked_item + '.'
        else:
            observation = "Nothing happens."
            reward = 0
            done = False
    # 
    if '\n' in observation:
        observation = '\n'+observation.strip()+'\n'
    observation = '=>' + observation + close_token
    return action, observation




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
    for resp_line in token_sequ.split('\n'):
        try:
            if '=' in resp_line and not '=>' in resp_line and not '<=' in resp_line:
                resp_line_variable=resp_line.split('=')[0].strip()
                exec('global '+resp_line_variable, globals())
        except Exception as e_resp:
            print(resp_line)
        try:
            exec(resp_line, globals())
        except Exception as e_resp:
            print('')

def get_action(resp_last_line_input):
    global resp_last_line
    global resp_last_line_for_action
    resp_last_line=resp_last_line_input
    variable_names = list(np.unique(re.findall(r"\{(.*?)\}", resp_last_line)))
    if len(variable_names)>0:
        replace_var_exec='resp_last_line_for_action=resp_last_line.format('
        for variable_name in variable_names:
            replace_var_exec=replace_var_exec+variable_name+'='+variable_name+','
        replace_var_exec=replace_var_exec[:-1]+')'
        try:
            exec(replace_var_exec, globals())
        except Exception as e_replace_var:
            print('')
    else:
        resp_last_line_for_action=resp_last_line
    action=resp_last_line_for_action.split('> ')[-1].strip()
    return action

def get_thread_status(token_sequ):
    if token_sequ[-1] == close_token or token_sequ[-1] == '\n' or '\nprint(' in token_sequ.split('\n')[-1]:
        return 'end'
    elif (token_sequ.split('\n')[-1].startswith('>') or token_sequ.split('\n')[-1].startswith('# >')) and not token_sequ.split('\n')[-1].strip().endswith('<='):
        return 'action'
    else:
        return 'spawn'






ob_task_list=[]
name_list=[]
reward_list=[]
for test_idx in range(test_set_size):
    if benchmark == 'webshop':
        action = 'reset'
        idx = f'fixed_{test_idx}'
        ob = env.step(idx, action)
        ob = ob[0].split('Instruction:  \n')[-1]
        ob = ob.split('\n[Search]')[0]
        ob = ob.strip()
        count_apostrophe=0
        for word in ob.split(' '):
            if word not in ["i'm", "that's", "aren't"]:
                if "'" in word:
                    count_apostrophe=count_apostrophe+1
        if count_apostrophe>0:
            prompt0_examples = prompt_d['thread_webshop'].replace("'",'"')
        ob = 'Instruction: '+ob
        # print(ob)
        # 
        ob_task = ob
        name = idx
        context0=ob
    # 
    print(ob_task + ' - ' + name)
    ob_task_list.append(ob_task)
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
    reward_list.append(reward)
    res_df=pd.DataFrame({'name':name_list,'task':ob_task_list,'reward':reward_list})
    print_res_df(res_df)



