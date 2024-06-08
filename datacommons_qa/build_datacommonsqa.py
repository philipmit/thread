
import os
import pandas as pd
import numpy as np
import requests 
import json 
import random
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt
import traceback
import time
import os



state_county_city_df = pd.read_csv('./us_cities_states_counties.csv',delimiter='|')


def city_to_state2(city):
    assert not state_county_city_df[state_county_city_df['City'] == city].empty , f"No matching state found for city: {city}" 
    state = ""
    state = state_county_city_df.loc[state_county_city_df['City'] == city, 'State full']
    # get most common value in state
    state = state.mode().iloc[0]
    return state

def county_to_state2(county): 
    county =  county.upper()
    assert not state_county_city_df[state_county_city_df['County'] == county].empty , f"No matching state found for county: {county}" 
    state = ""
    state = state_county_city_df.loc[state_county_city_df['County'] == county, 'State full']
    state = state.mode().iloc[0]
    return state


def collect_data_commons(query2, variable_text):
    url = f"https://datacommons.org/api/explore/detect?q={query2}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contextHistory": [],
        "dc": ""
    }
    response = requests.post(url, headers=headers, json=data)
    res_data = json.loads(response.text) 
    if "entities" not in list(res_data.keys()):
        return None
    if len(res_data["debug"]["places_detected"]) == 0 :
        return None
    entity = res_data['entities'][0] 
    url = "https://api.datacommons.org/v2/observation"
    params = {
    "key": "AIzaSyCTI4Xz-UW_G2Q2RfknhcfdAnTHq5X5XuI",
    "entity.dcids": entity, 
    "select": ["entity", "variable", "value", "date"],
    "variable.dcids": variable
    }
    response = requests.get(url, params=params)
    obs_data = json.loads(response.text)
    entity_data = obs_data["byVariable"][variable]["byEntity"][entity]
    if "orderedFacets" not in list(entity_data.keys()):
        return None
    seq_data = entity_data["orderedFacets"][0]['observations']
    # print('\n\n\n\n')
    return seq_data


def get_times(year_list):
    import max
    year_list = sorted(year_list)
    year1=random.choice(year_list[:max(1,len(year_list)//2)])
    time1=str(year1)
    year2=random.choice([x for x in year_list if x > year1+1])
    time2=str(year2)
    return time1,time2


city_list = list(set(state_county_city_df['City'].tolist())) 

state_list = list(set(state_county_city_df['State full'].tolist()))
state_list_sorted = state_list.sort()
county_list = list(set(state_county_city_df['County'].tolist()))

variable_df=pd.read_csv("./datacommons_variables.csv")


question_templates_all={'template1': 'Was the minimum value from {time1} to {time2} of {variable_text} lower in {entity1} or {entity2}?',
'template2': 'Was {variable_text} in {state1} in {time1} higher or lower than the national average?',
'template3': 'Was the average value from {time1} to {time2} of {variable_text} higher in {entity1} or {entity2}?',
'template4': 'In {entity1} was {variable_text} increasing or decreasing from {time1} to {time2}?',
'template5': 'Was the maximum value from {time1} to {time2} of {variable_text} higher in {entity1} or {entity2}?',
'template6': 'According to data from {time1} to {time2} is {variable_text} expected to be lower in {entity1} or {entity2} in {time3}?',
'template7': 'From {time1} to {time2} was the minimum value of {variable_text} higher in {entity1} or {entity2}?',
'template8': 'Compared to {entity2} was the maximum value from {time1} to {time2} of {variable_text} higher or lower in {entity1}?',
'template9': 'Compared to {entity2} was the minimum value from {time1} to {time2} of {variable_text} higher or lower in {entity1}?',
'template10': 'Compared to {entity2} was the average value from {time1} to {time2} of {variable_text} higher or lower in {entity1}?',
'template11': 'From {time1} to {time2} was the average value of {variable_text} lower in {entity1} or {entity2}?',
'template12': 'Compared to {entity2} was {variable_text} higher or lower in {entity1} in {time1}?',
'template13': 'Based on the data from {time1} to {time2} in {entity1} is {variable_text} expected to increase or decrease going forward?',
'template14': 'From {time1} to {time2} was {variable_text} increasing faster in {entity1} or {entity2}?',
'template15': 'Will {variable_text} be higher in {entity1} or {entity2} in {time3} based on the data from {time1} to {time2}?',
'template16': 'Was {variable_text} higher in {entity1} or {entity2} in {time1}?',
'template17': 'Using data from {time1} to {time2} in {entity1} is {variable_text} expected to be above or below {entity2} in {time3}?',
'template18': 'Was there a positive or negative correlation in {variable_text} in {entity1} and {entity2} from {time1} to {time2}?',
'template19': 'In {time1} was {variable_text} lower in {entity1} or {entity2}?',
'template20': 'From {time1} to {time2} was the maximum value of {variable_text} lower in {entity1} or {entity2}?'}

# randomly sort the question templates
question_templates_all = dict(sorted(question_templates_all.items(), key=lambda item: random.random()))


question_templates=question_templates_all
template_list=list(question_templates.keys())


question_variable_list=[]
question_variable_list_track=[]
question_variable_used_list=[]


question_list=[]
answer_list=[]
total_questions=160
question_df = pd.DataFrame({'question': question_list, 'dc_answer': answer_list})
while len(question_df) < total_questions:
    template_numb=random.choice(template_list)
    print(question_list)
    print(template_numb)
    count_tries=0
    while count_tries < 200:
        count_tries+=1
        try:
            question_template=question_templates[template_numb] 
            ##########################################
            if count_tries == 100:
                question_variable_list_track=[]
            variable_df_not_in_question_variable_list = variable_df[~variable_df['Variable_DCID'].isin(question_variable_list_track)]
            variable_list_not_in_question_variable_list = variable_df_not_in_question_variable_list['Variable_DCID'].tolist()
            variable_text_list_not_in_question_variable_list = variable_df_not_in_question_variable_list['Variable_Text'].tolist()
            variable_numb=random.choice(list(range(len(variable_list_not_in_question_variable_list)))) 
            variable=variable_list_not_in_question_variable_list[variable_numb] 
            variable_text=variable_text_list_not_in_question_variable_list[variable_numb]
            print(variable)
            time3 = random.choice(list(range(2025, 2041)))
            random_float=random.uniform(0,1)
            if random_float<0.33:
                entity1=random.choice(city_list) 
                entity1_state = city_to_state2(entity1)
                entity1 = entity1 + ' ' + entity1_state
            elif random_float<0.5:
                entity1=random.choice(county_list) 
                entity1_state=county_to_state2(entity1)
                entity1 = entity1 + ' ' + entity1_state
            else:
                entity1=random.choice(state_list)
            random_float=random.uniform(0,1)
            if random_float<0.33:
                entity2=random.choice(city_list) 
                entity2_state = city_to_state2(entity2)
                entity2 = entity2 + ' ' + entity2_state
            elif random_float<0.5:
                entity2=random.choice(county_list) 
                entity2_state=county_to_state2(entity2)
                entity2 = entity2 + ' ' + entity2_state
            else:
                entity2=random.choice(state_list)
            if entity1==entity2:
                continue
            if template_numb=='template4': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1])
                print(entity1)
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                if slope<0:
                    answer='decreasing'
                else:
                    answer='increasing'
                plt.figure(figsize=(10, 6))
                plt.scatter(X, y, label=variable_text.capitalize(), color="navy")
                plt.plot(X, intercept + slope * np.array(X), color="darkred", label="Trend line")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
                plt.xticks(rotation=45)  
                plt.tight_layout()
            elif template_numb=='template12' : 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print(f"E1: {entity1} NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print(f"E2: {entity2} NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                target_value_entity1=[x['value'] for x in seq_data_entity1 if x['date']==time1][0]
                target_value_entity2=[x['value'] for x in seq_data_entity2 if x['date']==time1][0]
                if target_value_entity1==target_value_entity2:
                    continue
                elif target_value_entity1<target_value_entity2:
                    answer='lower'
                else:
                    answer='higher'
                plt.figure(figsize=(10, 6))
                X = [entity1, entity2]
                y = [target_value_entity1, target_value_entity2]
                plt.bar(X, y)
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel('Locations')
                plt.ylabel(variable_text.capitalize())
            elif template_numb=='template16' : 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print(f"E1: {entity1} NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print(f"E2: {entity2} NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                target_value_entity1=[x['value'] for x in seq_data_entity1 if x['date']==time1][0]
                target_value_entity2=[x['value'] for x in seq_data_entity2 if x['date']==time1][0]
                if target_value_entity1==target_value_entity2:
                    continue
                elif target_value_entity1>target_value_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                X = [entity1, entity2]
                y = [target_value_entity1, target_value_entity2]
                plt.bar(X, y)
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel('Locations')
                plt.ylabel(variable_text.capitalize())
            elif template_numb=='template19' : 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print(f"E1: {entity1} NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print(f"E2: {entity2} NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                target_value_entity1=[x['value'] for x in seq_data_entity1 if x['date']==time1][0]
                target_value_entity2=[x['value'] for x in seq_data_entity2 if x['date']==time1][0]
                if target_value_entity1==target_value_entity2:
                    continue
                elif target_value_entity1<target_value_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                X = [entity1, entity2]
                y = [target_value_entity1, target_value_entity2]
                plt.bar(X, y)
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel('Locations')
                plt.ylabel(variable_text.capitalize())                
            elif template_numb=='template14': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X1=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y1=[x['value'] for x in seq_data_entity1_in_timeframe]
                slope_entity1, intercept1, r_value, p_value, std_err = stats.linregress(X1, y1)
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X2=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y2=[x['value'] for x in seq_data_entity2_in_timeframe]
                slope_entity2, intercept2, r_value, p_value, std_err = stats.linregress(X2, y2)
                if slope_entity1<=0 and slope_entity2<=0:
                    continue 
                if slope_entity1>slope_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.plot(X1, intercept1 + slope_entity1 * np.array(X1), color="darkred", label="Trend line")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.plot(X2, intercept2 + slope_entity2 * np.array(X2), color="darkred", label="Trend line")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb == 'template2': 
                if "Count" in variable:
                    continue
                state=random.choice(state_list)
                state = state
                query=variable_text+' in '+ state
                seq_data_entity1, variable_used=collect_data_commons(query)
                if seq_data_entity1 is None:
                    continue
                query=variable_text+' in '+ "United States"
                seq_data_entity2, variable_used=collect_data_commons(query)
                if seq_data_entity2 is None:
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                query=question_template.format(variable_text=variable_text,time1=time1, state1=state) 
                print(query)  
                target_value_entity1=[x['value'] for x in seq_data_entity1 if x['date']==time1][0]
                target_value_entity2=[x['value'] for x in seq_data_entity2 if x['date']==time1][0]
                if target_value_entity1 == target_value_entity2: 
                    continue
                elif target_value_entity1 < target_value_entity2:
                    answer = 'lower'
                else:
                    answer = 'higher'
                X = [state, "United States"]
                y = [target_value_entity1, target_value_entity2]
                plt.figure(figsize=(10, 6))
                plt.bar(X, y)
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel(f'{state} vs {"National Average"}')
                plt.ylabel(variable_text.capitalize())
            elif template_numb=='template10': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1, X1, y1 =np.mean(y), X, y
                #
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2, X2, y2 =np.mean(y), X, y
                #
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1>target_val_entity2:
                    answer='higher'
                else:
                    answer='lower'
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template3': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1, X1, y1 =np.mean(y), X, y
                #
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2, X2, y2 =np.mean(y), X, y
                #
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1>target_val_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template11': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1, X1, y1 =np.mean(y), X, y
                #
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2, X2, y2 =np.mean(y), X, y
                #
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1<target_val_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template9': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1=np.min(y)
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2=np.min(y)
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1>target_val_entity2:
                    answer='higher'
                else:
                    answer='lower'
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template7': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1=np.min(y)
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2=np.min(y)
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1>target_val_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template1': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1=np.min(y)
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2=np.min(y)
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1<target_val_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template5': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1=np.max(y)
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2=np.max(y)
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1>target_val_entity2:
                    answer='higher'
                else:
                    answer='lower'
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template8': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1=np.max(y)
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2=np.max(y)
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1>target_val_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template20': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                target_val_entity1=np.max(y)
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                target_val_entity2=np.max(y)
                if target_val_entity1==target_val_entity2:
                    continue 
                if target_val_entity1<target_val_entity2:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template18': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X1=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y1=[x['value'] for x in seq_data_entity1_in_timeframe]
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X2=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y2=[x['value'] for x in seq_data_entity2_in_timeframe]
                corr, _ = pearsonr(y1, y2)
                if corr==0:
                    continue 
                if corr<0:
                    answer='negative'
                else:
                    answer='positive'
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template13': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                if len(seq_data_entity1)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                slope_entity1=slope
                if slope_entity1==0:
                    continue 
                if slope_entity1<0:
                    answer='decrease'
                else:
                    answer='increase'
                plt.figure(figsize=(10, 6))
                plt.scatter(X, y, label=entity1, color="navy")
                plt.plot(X, intercept + slope_entity1 * np.array(X), color="darkred", label="Trend line")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template17': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,time3=time3, entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                slope_entity1, X1, y1, intercept1 =slope, X, y, intercept
                estimated_value1_time3 = slope*time3+intercept
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                slope_entity2, X2, y2, intercept2 =slope, X, y, intercept
                estimated_value2_time3 = slope*time3+intercept
                if estimated_value1_time3==estimated_value2_time3:
                    continue 
                if estimated_value1_time3>estimated_value2_time3:
                    answer='above'
                else:
                    answer='below'
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.plot(X1, intercept1 + slope_entity1 * np.array(X1), color="darkred", label="Trend line")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.plot(X2, intercept2 + slope_entity2 * np.array(X2), color="darkred", label="Trend line")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template15': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,time3=time3, entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                slope_entity1, X1, y1, intercept1 =slope, X, y, intercept
                estimated_value1_time3 = slope*time3+intercept
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                slope_entity2, X2, y2, intercept2 =slope, X, y, intercept
                estimated_value2_time3 = slope*time3+intercept
                if estimated_value1_time3==estimated_value2_time3:
                    continue 
                if estimated_value1_time3>estimated_value2_time3:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.plot(X1, intercept1 + slope_entity1 * np.array(X1), color="darkred", label="Trend line")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.plot(X2, intercept2 + slope_entity2 * np.array(X2), color="darkred", label="Trend line")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            elif template_numb=='template6': 
                query2=variable_text+' in '+ entity1
                seq_data_entity1, variable_used=collect_data_commons(query2)
                if seq_data_entity1 is None:
                    print("E1 NOT FOUND")
                    continue
                query2=variable_text+' in '+ entity2
                seq_data_entity2, variable_used=collect_data_commons(query2)
                if seq_data_entity2 is None:
                    print("E2 NOT FOUND")
                    continue
                time1, time2 = get_times([int(x['date']) for x in seq_data_entity1 if x['date'] in [x['date'] for x in seq_data_entity2]])
                if len(seq_data_entity1)<2 or len(seq_data_entity2)<2:
                    continue
                query=question_template.format(variable_text=variable_text,time1=time1,time2=time2,time3=time3, entity1=entity1,entity2=entity2)
                print(query)
                seq_data_entity1_in_timeframe=[x for x in seq_data_entity1 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity1_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity1_in_timeframe]
                y=[x['value'] for x in seq_data_entity1_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                slope_entity1, X1, y1, intercept1 =slope, X, y, intercept
                estimated_value1_time3 = slope*time3+intercept
                seq_data_entity2_in_timeframe=[x for x in seq_data_entity2 if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
                if len(seq_data_entity2_in_timeframe)<2:
                    continue
                X=[int(x['date']) for x in seq_data_entity2_in_timeframe]
                y=[x['value'] for x in seq_data_entity2_in_timeframe]
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                slope_entity2, X2, y2, intercept2 =slope, X, y, intercept
                estimated_value2_time3 = slope*time3+intercept
                if estimated_value1_time3==estimated_value2_time3:
                    continue 
                if estimated_value1_time3<estimated_value2_time3:
                    answer=entity1
                else:
                    answer=entity2
                plt.figure(figsize=(10, 6))
                plt.scatter(X1, y1, label=entity1, color="navy")
                plt.plot(X1, intercept1 + slope_entity1 * np.array(X1), color="darkred", label="Trend line")
                plt.scatter(X2, y2, label=entity2, color="green")
                plt.plot(X2, intercept2 + slope_entity2 * np.array(X2), color="darkred", label="Trend line")
                plt.title(query2+'\nAnswer: '+answer)
                plt.xlabel("Year")
                plt.ylabel(variable_text.capitalize())
                plt.legend()
            answer_list.append(answer)
            question_list.append(query)
            question_variable_list.append(variable)
            question_variable_list_track.append(variable)
            question_variable_used_list.append(variable_used)
            print(np.unique(question_variable_list))
            print(len(np.unique(question_variable_list)))
            question_df = pd.DataFrame({'question': question_list, 'answer': answer_list})
            question_df
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            continue
    # plt.savefig(f'./figures/figure_{len(question_df)}.pdf')
    question_df = pd.DataFrame({'question': question_list, 'dc_answer': answer_list})
    question_df
    print('\n\n\n')
    print(question_df)
    import time
    time.sleep(5)





