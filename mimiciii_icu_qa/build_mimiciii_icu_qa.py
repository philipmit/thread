import io
import time
import re
import gc
import os
import yaml
import sys
import numpy as np
import pandas as pd
import json

import requests  
import random
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import traceback
import time

pd.set_option('max_colwidth', 10000)
pd.set_option('display.max_rows', 1000)



# You first need to build the in-hospital-mortality dataset following the steps outlined here: https://github.com/YerevaNN/mimic3-benchmarks
# Once this is complete, you should have a folder labeled in-hospital-mortality. Please set the MIMIC_DATA_DIR variable to the path of this folder.
# MIMIC_DATA_DIR = ...




question_templates_all={
    'template1':'Was the {variable_text1} increasing or decreasing for patient {patient1} from hour {time1} to {time2}?',
    'template2':'Compared to patient {patient2}, was the maximum {variable_text1} higher or lower for patient {patient1}?',
    'template3':'Was the minimum {variable_text1} higher or lower for patient {patient1} compared to patient {patient2}?',
    'template4':'For patient {patient1}, was the average {variable_text1} higher or lower compared to patient {patient2}?',
    'template5':'Was the {variable_text1} increasing faster for patient {patient1} or patient {patient2} during the first {time1} hours of their ICU stay?',
    'template6':'Was the {variable_text1} decreasing faster for patient {patient1} or patient {patient2} during the first {time1} hours of their ICU stay?',
    'template7':'Was the average {variable_text1} of patient {patient1} from hour {time1} to {time2} higher or lower than the average among all patients who are within 5 years of age of this patient?', 
    'template8':'Compard to the average among all patients who did not expire in the hospital, was the average {variable_text1} of patient {patient1} from hour {time1} to {time2} higher or lower?', 
    'template9':'Was the average {variable_text1} of patient {patient1} from hour {time1} to {time2} higher or lower than the average among all patients who expired in the hospital?', 
    'template10':'Compared to the average among all patients with {phenotype_text1}, was the average {variable_text1} of patient {patient1} from hour {time1} to {time2} higher or lower?', 
    'template11':'From hour {time1} to {time2}, was there a positive or negative correlation between {variable_text1} and {variable_text2} in patient {patient1}?',
    'template12':'Was the correlation between {variable_text1} and {variable_text2} in patient {patient1} higher or lower than the correlation between {variable_text1} and {variable_text3} from hour {time1} to {time2}?',
    'template13': 'Between hour {time1} and {time2}, what time was {variable_text1} the highest for patient {patient1}?',
    'template14': 'At what time did {variable_text1} reach its lowest value for patient {patient1} between hour {time1} and {time2}?',
    'template15': 'The {variable_text1} for patient {patient1} increased the most at what time?',
    'template16': 'For patient {patient1}, at what timepoint did {variable_text1} decrease the most?',
    'template17': 'Is {variable_text1} expected to increase or decrease going forward for patient {patient1} based on the data from hour {time1} to {time2}?',
    'template18': 'For patient {patient1}, is {variable_text1} expected to be above or below the average {variable_text1} among patients with {phenotype_text1} at hour {time3} based on the data from hour {time1} to {time2}?',
    'template19': 'Based on the data from hour {time1} to {time2}, is {variable_text1} for patient {patient1} expected to be above or below the highest average {variable_text1} among patients with {phenotype_text1} at hour {time3}?',
    'template20': 'Is {variable_text1} for patient {patient1} expected to be above or below the lowest average {variable_text1} among patients with {phenotype_text1} at hour {time3} based on the data from hour {time1} to {time2}?'

}



variables = {
    'Capillary refill rate': 'capillary refill rate',
    'Diastolic blood pressure': 'diastolic blood pressure',
    'Fraction inspired oxygen': 'fraction inspired oxygen',
    'Glucose': 'blood sugar',
    'Heart Rate': 'heart rate',
    'Mean blood pressure': 'mean blood pressure',
    'Oxygen saturation': 'oxygen saturation',
    'Respiratory rate': 'respiratory rate',
    'Systolic blood pressure': 'systolic blood pressure',
    'Temperature': 'temperature',
    'Weight': 'weight',
    'pH': 'pH'
}
variable_text_list = [variables[x] for x in list(variables.keys())]
variable_text_col_list = list(variables.keys())




phenotypes = {
    'Acute cerebrovascular disease': 'acute cerebrovascular disease',
    'Acute myocardial infarction': 'acute myocardial infarction',
    'Cardiac dysrhythmias': 'dysrhythmias',
    'Chronic kidney disease': 'chronic kidney disease',
    'Chronic obstructive pulmonary disease and bronchiectasis': 'chronic obstructive pulmonary disease',
    'Conduction disorders': 'conduction disorders',
    'Coronary atherosclerosis and other heart disease': 'atherosclerosis',
    'Diabetes mellitus with complications': 'diabetes',
    'Disorders of lipid metabolism': 'disorders of lipid metabolism',
    'Gastrointestinal hemorrhage': 'gastrointestinal hemorrhage',
    'Hypertension with complications and secondary hypertension': 'hypertension with complications',
    'Pleurisy; pneumothorax; pulmonary collapse': 'pulmonary collapse',
    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)': 'pneumonia',
    'Respiratory failure; insufficiency; arrest (adult)': 'respiratory failure',
    'Septicemia (except in labor)': 'septicemia'
}
phenotype_text_list = [phenotypes[x] for x in list(phenotypes.keys())]
phenotype_text_col_list = list(phenotypes.keys())




patient_IDs_train=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/train/listfile.csv")
patient_IDs_test=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/test/listfile.csv")
patient_IDs_all = pd.concat([patient_IDs_train, patient_IDs_test])
patient_IDs=list(patient_IDs_all['stay'])
patient_IDs=[x.split('_')[0] for x in patient_IDs]
patient_IDs=list(np.unique(patient_IDs))
patient_IDs
len(patient_IDs)
patient_IDs=patient_IDs[1:]


# copy all patients in train and test to /in-hospital-mortality/all/
os.system('mkdir '+ MIMIC_DATA_DIR + "/in-hospital-mortality/all/")
os.system('cp '+ MIMIC_DATA_DIR + "/in-hospital-mortality/train/* "+ MIMIC_DATA_DIR + "/in-hospital-mortality/all/")
os.system('cp '+ MIMIC_DATA_DIR + "/in-hospital-mortality/test/* "+ MIMIC_DATA_DIR + "/in-hospital-mortality/all/")


question_templates=question_templates_all
template_list=list(question_templates.keys())


random.seed(0)
question_list=[]
answer_list=[]
total_questions=160
while len(question_df) < total_questions:
    template_numb=random.choice(template_list)
    print(question_list)
    print(template_numb)
    try:
        print('\n\n\n\n*********************************************************************************')
        print(len(question_list)) # prints number of questions which have been generated
        # 
        patient1=random.choice(patient_IDs)
        patient2=patient1
        while (patient2==patient1):
            patient2=random.choice(patient_IDs)
        patient3=patient1
        while (patient3 in [patient1, patient2]):
            patient3=random.choice(patient_IDs)
        variable_numb_list=list(range(len(variable_text_list)))
        variable_numb1=random.choice(variable_numb_list)
        variable_numb2=random.choice([x for x in variable_numb_list if not x==variable_numb1])
        variable_numb3=random.choice([x for x in variable_numb_list if not x in [variable_numb1, variable_numb2]])
        phenotype_numb1=random.choice(list(range(len(phenotype_text_list))))
        phenotype_text1=phenotype_text_list[phenotype_numb1]
        phenotype_text_col1=phenotype_text_col_list[phenotype_numb1]
        variable_text1=variable_text_list[variable_numb1]
        variable_text_col1=variable_text_col_list[variable_numb1]
        variable_text2=variable_text_list[variable_numb2]
        variable_text_col2=variable_text_col_list[variable_numb2]
        variable_text3=variable_text_list[variable_numb3]
        variable_text_col3=variable_text_col_list[variable_numb3]
        time1 = random.randint(1, 20)
        time2 = random.randint(time1+1, 46)
        time3 = random.randint(time2+1, 48)
        N = random.choice(range(2, 6))
        print('patient1: '+ str(patient1))
        print('patient2: '+ str(patient2))
        print('time1: '+ str(time1))
        print('time2: '+ str(time2))
        print('variable_text1: '+ str(variable_text1))
        print('variable_text2: '+ str(variable_text2))
        print('variable_text3: '+ str(variable_text3))
        question_template=question_templates[template_numb] 
        query = question_template.format(variable_text1=variable_text1, N=N,variable_text2=variable_text2, variable_text3=variable_text3, time1=time1, time2=time2, time3=time3, patient1=patient1, patient2=patient2, patient3=patient3, phenotype_text1=phenotype_text1)
        print()
        if template_numb=='template1': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            if slope==0:
                continue
            if slope<0:
                answer='decreasing'
            else:
                answer='increasing'
        elif template_numb=='template2' : 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient2_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient2)+"_episode1_timeseries.csv")
            patient2_df = patient2_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient2_df = patient2_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2 or len(patient2_df)<=2:
                continue
            target_value_patient1=max(list(patient1_df[variable_text_col1]))
            target_value_patient2=max(list(patient2_df[variable_text_col1]))
            if target_value_patient1==target_value_patient2:
                continue
            elif target_value_patient1<target_value_patient2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb=='template3': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient2_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient2)+"_episode1_timeseries.csv")
            patient2_df = patient2_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient2_df = patient2_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2 or len(patient2_df)<=2:
                continue
            target_value_patient1=np.mean(list(patient1_df[variable_text_col1]))
            target_value_patient2=np.mean(list(patient2_df[variable_text_col1]))
            if target_value_patient1==target_value_patient2:
                continue
            elif target_value_patient1<target_value_patient2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb=='template4': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient2_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient2)+"_episode1_timeseries.csv")
            patient2_df = patient2_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient2_df = patient2_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2 or len(patient2_df)<=2:
                continue
            target_value_patient1=min(list(patient1_df[variable_text_col1]))
            target_value_patient2=min(list(patient2_df[variable_text_col1]))
            if target_value_patient1==target_value_patient2:
                continue
            elif target_value_patient1<target_value_patient2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb == "template5": 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient2_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient2)+"_episode1_timeseries.csv")
            patient2_df = patient2_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient2_df = patient2_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1)]
            seq_data_patient2_in_timeframe_df = patient2_df[(patient2_df['Hours'] >= time1)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope_patient1, intercept_patient1, r_value_patient1, p_value_patient1, std_err_patient1 = stats.linregress(X, y)
            X=list(seq_data_patient2_in_timeframe_df['Hours'])
            y=list(seq_data_patient2_in_timeframe_df[variable_text_col1])
            slope_patient2, intercept_patient2, r_value_patient2, p_value_patient2, std_err_patient2 = stats.linregress(X, y)
            if slope_patient1<=0 and slope_patient2<=0:
                continue
            if slope_patient1==slope_patient2:
                continue
            if slope_patient1>slope_patient2:
                answer=patient1
            else:
                answer=patient2
        elif template_numb == "template6": 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient2_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient2)+"_episode1_timeseries.csv")
            patient2_df = patient2_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient2_df = patient2_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1)]
            seq_data_patient2_in_timeframe_df = patient2_df[(patient2_df['Hours'] >= time1)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope_patient1, intercept_patient1, r_value_patient1, p_value_patient1, std_err_patient1 = stats.linregress(X, y)
            X=list(seq_data_patient2_in_timeframe_df['Hours'])
            y=list(seq_data_patient2_in_timeframe_df[variable_text_col1])
            slope_patient2, intercept_patient2, r_value_patient2, p_value_patient2, std_err_patient2 = stats.linregress(X, y)
            if slope_patient1>=0 and slope_patient2>=0:
                continue
            if slope_patient1==slope_patient2:
                continue
            if slope_patient1<slope_patient2:
                answer=patient1
            else:
                answer=patient2
        elif template_numb == "template7": 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            all_stays = pd.read_csv(MIMIC_DATA_DIR + "/all_stays.csv")
            all_stays.shape
            patient1_age = all_stays['AGE'][all_stays['SUBJECT_ID']==int(patient1)].values[0]
            patients_to_compare = all_stays[(all_stays['AGE']>=patient1_age-5) & (all_stays['AGE']<=patient1_age+5)]['SUBJECT_ID']
            patients_to_compare = np.unique(patients_to_compare)
            len(patients_to_compare)
            # get average variable_text1 across all patients within patients_to_compare
            # initialize list with 7144 elements to store target values
            target_value_patient_list = [None]*len(patients_to_compare)
            print('start for loop')
            for patient_i in range(len(patients_to_compare)):
                try:
                    patient_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patients_to_compare[patient_i])+"_episode1_timeseries.csv")
                    patient_df = patient_df[['Hours', variable_text_col1]]
                    # remove rows where variable_text is NaN
                    patient_df = patient_df.dropna(subset=[variable_text_col1])
                    if len(patient_df)<=2:
                        continue
                    target_value_patient=np.mean(list(patient_df[variable_text_col1]))
                    target_value_patient_list[patient_i] = target_value_patient
                except Exception as e:
                    continue
            print('end for loop')
            # remove None values from target_value_patient_list
            target_value_patient_list = [x for x in target_value_patient_list if x is not None]
            target_value_patient1=np.mean(list(patient1_df[variable_text_col1]))
            target_value2=np.mean(target_value_patient_list)
            if target_value_patient1==target_value2:
                continue
            elif target_value_patient1<target_value2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb == "template8": 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            all_stays = pd.read_csv(MIMIC_DATA_DIR + "/all_stays.csv")
            all_stays.shape
            patients_to_compare = all_stays[all_stays['MORTALITY_INHOSPITAL']==0]['SUBJECT_ID']
            patients_to_compare = np.unique(patients_to_compare)
            len(patients_to_compare)
            # get average variable_text1 across all patients within patients_to_compare
            # initialize list with 7144 elements to store target values
            target_value_patient_list = [None]*len(patients_to_compare)
            for patient_i in range(len(patients_to_compare)):
                try:
                    patient_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patients_to_compare[patient_i])+"_episode1_timeseries.csv")
                    patient_df = patient_df[['Hours', variable_text_col1]]
                    # remove rows where variable_text is NaN
                    patient_df = patient_df.dropna(subset=[variable_text_col1])
                    if len(patient_df)<=2:
                        continue
                    target_value_patient=np.mean(list(patient_df[variable_text_col1]))
                    target_value_patient_list[patient_i] = target_value_patient
                except Exception as e:
                    continue
            # remove None values from target_value_patient_list
            target_value_patient_list = [x for x in target_value_patient_list if x is not None]
            target_value_patient1=np.mean(list(patient1_df[variable_text_col1]))
            target_value2=np.mean(target_value_patient_list)
            if target_value_patient1==target_value2:
                continue
            elif target_value_patient1<target_value2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb == "template9": 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            all_stays = pd.read_csv(MIMIC_DATA_DIR + "/all_stays.csv")
            all_stays.shape
            patients_to_compare = all_stays[all_stays['MORTALITY_INHOSPITAL']==1]['SUBJECT_ID']
            patients_to_compare = np.unique(patients_to_compare)
            len(patients_to_compare)
            # get average variable_text1 across all patients within patients_to_compare
            # initialize list with 7144 elements to store target values
            target_value_patient_list = [None]*len(patients_to_compare)
            for patient_i in range(len(patients_to_compare)):
                try:
                    patient_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patients_to_compare[patient_i])+"_episode1_timeseries.csv")
                    patient_df = patient_df[['Hours', variable_text_col1]]
                    # remove rows where variable_text is NaN
                    patient_df = patient_df.dropna(subset=[variable_text_col1])
                    if len(patient_df)<=2:
                        continue
                    target_value_patient=np.mean(list(patient_df[variable_text_col1]))
                    target_value_patient_list[patient_i] = target_value_patient
                except Exception as e:
                    continue
            # remove None values from target_value_patient_list
            target_value_patient_list = [x for x in target_value_patient_list if x is not None]
            target_value_patient1=np.mean(list(patient1_df[variable_text_col1]))
            target_value2=np.mean(target_value_patient_list)
            if target_value_patient1==target_value2:
                continue
            elif target_value_patient1<target_value2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb == "template10": 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            pheno=pd.read_csv(MIMIC_DATA_DIR + "/phenotype_labels.csv")
            # list(pheno.columns)
            all_stays = pd.read_csv(MIMIC_DATA_DIR + "/all_stays.csv")
            all_stays.shape
            # list(all_stays.columns)
            # combine all_stays and pheno columnwise
            all_stays = pd.concat([all_stays, pheno], axis=1)
            all_stays.shape
            # list(all_stays.columns)
            patients_to_compare = all_stays[all_stays[phenotype_text_col1]==1]['SUBJECT_ID']
            patients_to_compare = np.unique(patients_to_compare)
            len(patients_to_compare)
            # get average variable_text1 across all patients within patients_to_compare
            # initialize list with 7144 elements to store target values
            target_value_patient_list = [None]*len(patients_to_compare)
            for patient_i in range(len(patients_to_compare)):
                try:
                    patient_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patients_to_compare[patient_i])+"_episode1_timeseries.csv")
                    patient_df = patient_df[['Hours', variable_text_col1]]
                    # remove rows where variable_text is NaN
                    patient_df = patient_df.dropna(subset=[variable_text_col1])
                    if len(patient_df)<=2:
                        continue
                    target_value_patient=np.mean(list(patient_df[variable_text_col1]))
                    target_value_patient_list[patient_i] = target_value_patient
                except Exception as e:
                    continue
            # remove None values from target_value_patient_list
            target_value_patient_list = [x for x in target_value_patient_list if x is not None]
            target_value_patient1=np.mean(list(patient1_df[variable_text_col1]))
            target_value2=np.mean(target_value_patient_list)
            if target_value_patient1==target_value2:
                continue
            elif target_value_patient1<target_value2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb=='template11': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1, variable_text_col2]]
            patient1_df = patient1_df.dropna(subset=[variable_text_col1, variable_text_col2])
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            # get correlation between variable_text1 and variable_text2
            var1=list(patient1_df[variable_text_col1])
            var2=list(patient1_df[variable_text_col2])
            corr, _ = pearsonr(var1, var2)
            if corr==0:
                continue
            if corr<0:
                answer='negative'
            else:
                answer='positive'
        elif template_numb=='template12': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1, variable_text_col2, variable_text_col3]]
            patient1_df = patient1_df.dropna(subset=[variable_text_col1, variable_text_col2, variable_text_col3])
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            # get correlation between variable_text1 and variable_text2
            var1=list(patient1_df[variable_text_col1])
            var2=list(patient1_df[variable_text_col2])
            var3=list(patient1_df[variable_text_col3])
            corr1, _ = pearsonr(var1, var2)
            corr2, _ = pearsonr(var1, var3)
            if corr1==corr2 :
                continue
            if corr1<corr2:
                answer='lower'
            else:
                answer='higher'
        elif template_numb=='template13': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            # filter based on time1 and time2
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            # get time when variable_text_col1 was the highest
            idx_target = list(patient1_df[variable_text_col1]).index(max(list(patient1_df[variable_text_col1])))
            hour_target = patient1_df['Hours'].iloc[idx_target]
            answer=hour_target
        elif template_numb=='template14': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            # filter based on time1 and time2
            patient1_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            if len(patient1_df)<=2:
                continue
            # get time when variable_text_col1 was the highest
            idx_target = list(patient1_df[variable_text_col1]).index(min(list(patient1_df[variable_text_col1])))
            hour_target = patient1_df['Hours'].iloc[idx_target]
            answer=hour_target
        elif template_numb=='template15':
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=5:
                continue
            var_diff=[]
            for i in range(1,len(patient1_df)):
                var_diff.append(patient1_df[variable_text_col1].iloc[i]-patient1_df[variable_text_col1].iloc[i-1])
            idx_target = var_diff.index(max(var_diff))
            hour_target = list(patient1_df['Hours'])[idx_target]
            answer=hour_target
        elif template_numb=='template16':
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=5:
                continue
            var_diff=[]
            for i in range(1,len(patient1_df)):
                var_diff.append(patient1_df[variable_text_col1].iloc[i]-patient1_df[variable_text_col1].iloc[i-1])
            idx_target = var_diff.index(min(var_diff))
            hour_target = list(patient1_df['Hours'])[idx_target]
            answer=hour_target
        elif template_numb=='template17': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            # 'Based on the data from hour {time1} to {time2} for patient {patient1}, is {variable_text1} expected to increase or decrease?',
            if slope==0:
                continue
            if slope>0:
                answer='increase'
            else:
                answer='decrease'
        elif template_numb=='template18': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            estimated_value_time3 = slope*time3+intercept
            # 
            pheno=pd.read_csv(MIMIC_DATA_DIR + "/phenotype_labels.csv")
            # list(pheno.columns)
            all_stays = pd.read_csv(MIMIC_DATA_DIR + "/all_stays.csv")
            all_stays.shape
            # list(all_stays.columns)
            # combine all_stays and pheno columnwise
            all_stays = pd.concat([all_stays, pheno], axis=1)
            all_stays.shape
            # list(all_stays.columns)
            patients_to_compare = all_stays[all_stays[phenotype_text_col1]==1]['SUBJECT_ID']
            patients_to_compare = np.unique(patients_to_compare)
            len(patients_to_compare)
            # get average variable_text1 across all patients within patients_to_compare
            target_value_patient_list = [None]*len(patients_to_compare)
            for patient_i in range(len(patients_to_compare)):
                try:
                    patient_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patients_to_compare[patient_i])+"_episode1_timeseries.csv")
                    patient_df = patient_df[['Hours', variable_text_col1]]
                    # remove rows where variable_text is NaN
                    patient_df = patient_df.dropna(subset=[variable_text_col1])
                    if len(patient_df)<=2:
                        continue
                    target_value_patient=np.mean(list(patient_df[variable_text_col1]))
                    target_value_patient_list[patient_i] = target_value_patient
                except Exception as e:
                    continue
            # remove None values from target_value_patient_list
            target_value_patient_list = [x for x in target_value_patient_list if x is not None]
            target_value2=np.mean(target_value_patient_list)
            # 
            if estimated_value_time3==target_value2:
                continue
            if estimated_value_time3>target_value2:
                answer='above'
            else:
                answer='below'
        elif template_numb=='template19': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            estimated_value_time3 = slope*time3+intercept
            # 
            pheno=pd.read_csv(MIMIC_DATA_DIR + "/phenotype_labels.csv")
            # list(pheno.columns)
            all_stays = pd.read_csv(MIMIC_DATA_DIR + "/all_stays.csv")
            all_stays.shape
            # list(all_stays.columns)
            # combine all_stays and pheno columnwise
            all_stays = pd.concat([all_stays, pheno], axis=1)
            all_stays.shape
            # list(all_stays.columns)
            patients_to_compare = all_stays[all_stays[phenotype_text_col1]==1]['SUBJECT_ID']
            patients_to_compare = np.unique(patients_to_compare)
            len(patients_to_compare)
            # get average variable_text1 across all patients within patients_to_compare
            target_value_patient_list = [None]*len(patients_to_compare)
            for patient_i in range(len(patients_to_compare)):
                try:
                    patient_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patients_to_compare[patient_i])+"_episode1_timeseries.csv")
                    patient_df = patient_df[['Hours', variable_text_col1]]
                    # remove rows where variable_text is NaN
                    patient_df = patient_df.dropna(subset=[variable_text_col1])
                    if len(patient_df)<=2:
                        continue
                    target_value_patient=np.mean(list(patient_df[variable_text_col1]))
                    target_value_patient_list[patient_i] = target_value_patient
                except Exception as e:
                    continue
            # remove None values from target_value_patient_list
            target_value_patient_list = [x for x in target_value_patient_list if x is not None]
            target_value2=max(target_value_patient_list)
            # 
            if estimated_value_time3==target_value2:
                continue
            if estimated_value_time3>target_value2:
                answer='above'
            else:
                answer='below'
        elif template_numb=='template20': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            estimated_value_time3 = slope*time3+intercept
            # 
            pheno=pd.read_csv(MIMIC_DATA_DIR + "/phenotype_labels.csv")
            # list(pheno.columns)
            all_stays = pd.read_csv(MIMIC_DATA_DIR + "/all_stays.csv")
            all_stays.shape
            # list(all_stays.columns)
            # combine all_stays and pheno columnwise
            all_stays = pd.concat([all_stays, pheno], axis=1)
            all_stays.shape
            # list(all_stays.columns)
            patients_to_compare = all_stays[all_stays[phenotype_text_col1]==1]['SUBJECT_ID']
            patients_to_compare = np.unique(patients_to_compare)
            len(patients_to_compare)
            # get average variable_text1 across all patients within patients_to_compare
            target_value_patient_list = [None]*len(patients_to_compare)
            for patient_i in range(len(patients_to_compare)):
                try:
                    patient_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patients_to_compare[patient_i])+"_episode1_timeseries.csv")
                    patient_df = patient_df[['Hours', variable_text_col1]]
                    # remove rows where variable_text is NaN
                    patient_df = patient_df.dropna(subset=[variable_text_col1])
                    if len(patient_df)<=2:
                        continue
                    target_value_patient=np.mean(list(patient_df[variable_text_col1]))
                    target_value_patient_list[patient_i] = target_value_patient
                except Exception as e:
                    continue
            # remove None values from target_value_patient_list
            target_value_patient_list = [x for x in target_value_patient_list if x is not None]
            target_value2=min(target_value_patient_list)
            # 
            if estimated_value_time3==target_value2:
                continue
            if estimated_value_time3>target_value2:
                answer='above'
            else:
                answer='below'
        elif template_numb=='template49': 
            patient1_df=pd.read_csv(MIMIC_DATA_DIR + "/in-hospital-mortality/all/"+str(patient1)+"_episode1_timeseries.csv")
            patient1_df = patient1_df[['Hours', variable_text_col1]]
            # remove rows where variable_text is NaN
            patient1_df = patient1_df.dropna(subset=[variable_text_col1])
            if len(patient1_df)<=2:
                continue
            # get values between time1 and time2
            seq_data_patient1_in_timeframe_df = patient1_df[(patient1_df['Hours'] >= time1) & (patient1_df['Hours'] <= time2)]
            X=list(seq_data_patient1_in_timeframe_df['Hours'])
            y=list(seq_data_patient1_in_timeframe_df[variable_text_col1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            # 'Based on the data from hour {time1} to {time2} for patient {patient1}, is {variable_text1} expected to increase or decrease?',
            if slope==0:
                continue
            if slope>0:
                answer='increase'
            else:
                answer='decrease'
        # 
        answer_list.append(answer)
        question_list.append(query)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        continue
    question_df = pd.DataFrame({'question': question_list, 'mimic_answer': answer_list})
    question_df
    question_df.to_csv('./mimiciii_icu_qa_benchmark3.txt',sep='\t')
    print('\n\n\n')
    print(question_df)
    import time
    time.sleep(5)



