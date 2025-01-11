from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import *

from numpy.random import default_rng
import os
from scipy.stats import  binom
import matplotlib.pyplot as plt
import click
import numpy as np
import torch
import wandb
import pandas as pd
from accelerate import Accelerator
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils.argument import ScriptArguments
from utils.data import (
    RewardDataCollatorWithPadding,
    build_dataset,
    post_filter_by_ratio,
    get_data,
)
from utils.trainer import (
    BTTRewardTrainer,
    RewardTrainer,
    RewardTrainerWithOracleCE,
    RewardTrainerWithRingeMargin,
    compute_CE_oracle,
    compute_ML_oracle,
)

from calibration_module.utils import compute_binary_score,compute_calibration_summary
from calibration_module.calibrator import HistogramCalibrator

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set for all GPUs
    set_seed(seed)  # Hugging Face's Trainer consistency

    # Ensure deterministic behavior across multi-GPU environments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def rej_prob_simulator(
    empirical_data,
    T_func,
    R_func,
    num_experiments,
    num_sampled_data,
    use_log,
    r):

    # Step 1: Generate all random samples at once (vectorized)
    n_data = len(empirical_data)
    rng = np.random.default_rng()
    sample_indices = rng.integers(low=0, high=n_data, size=(num_experiments, num_sampled_data))
        
    # shape = (num_experiments, sample_size)
    D_samples = empirical_data[sample_indices]



    t_values = T_func(D_samples)
    R_values = R_func(D_samples, r)


    # membership is a boolean array of shape (num_experiments,)
    membership = (t_values >= R_values[:, 0]) & (t_values <= R_values[:, 1])

    # Probability estimate is average of membership
    prob_est = membership.mean()
    if not use_log:
        return -prob_est
    else:
        return np.log(1-prob_est)



def rej_prob(mode, original_preference,  num_sampled_data,use_log,r):
    if mode=='self':
        p_mean=(1+r**2)/2
    else:
        #check if all original_preference>1/2
        if not np.all(original_preference>=1/2):
            raise ValueError("All original preference should be greater than 1/2")
    
        p_mean=np.mean(original_preference)*r+(1-r)/2
    

    
    prob=binom.cdf(num_sampled_data* p_mean, num_sampled_data, p_mean)
    if use_log:
        return np.log(1-prob)
    else:
        return -prob


def reg_prob_derivative(mode, r, original_preference, T_func, R_func, num_experiments=10000, num_sampled_data=1000, use_log=True,h=1e-6):
    """
    Compute the derivative of the rejection probability with respect to r.
    """
    
    empirical_data=agent_data_simulator(original_preference,r, mode)
    v1=rej_prob_simulator(empirical_data, T_func, R_func, num_experiments, num_sampled_data,use_log,r)
    empirical_data=agent_data_simulator(original_preference,r+h, mode)
    v2=rej_prob_simulator(empirical_data, T_func, R_func, num_experiments, num_sampled_data,use_log,r+h)
    dev=(v2-v1)/h


    return dev




def reg_prob_derivative_exact(mode, r, original_preference, T_func, R_func, num_experiments=10000, num_sampled_data=1000, use_log=True,h=5e-7):
    """
    Compute the derivative of the rejection probability with respect to r.
    """
    
   
    v1=rej_prob(mode, original_preference, num_sampled_data,use_log,r)
    v2=rej_prob(mode, original_preference, num_sampled_data,use_log,r+h)
    dev=(v2-v1)/(h)
    return dev
 


def agent_data_simulator(original_preference,r, mode=None):
    length=len(original_preference)
    
    if mode=='self':
        mean_values=(1+r**2)/2
    else:
        #check if all original_preference>1/2
        if not np.all(original_preference>=1/2):
            raise ValueError("All original preference should be greater than 1/2")
        mean_values=original_preference*r+(1-r)/2

    
    #generate bernoulli samples
    samples=np.random.binomial(1,mean_values,length)
    return samples



def T_create(data_batch):
        # data_batch.mean(axis=1) => shape (num_experiments,)
        return data_batch.mean(axis=1)

def R_create(Ep, mode, data_batch, r):
        # so shape => (num_experiments, 2)
        num_experiments = data_batch.shape[0]
        if mode =='self':
            high=(1+r**2)/2
        else:
            high=Ep*r+(1-r)/2
        high=high
        low=0
        # Create a 2D array of repeated intervals
        intervals = np.column_stack((np.full(num_experiments, low),
                                     np.full(num_experiments, high)))
        return intervals    


def get_incentive(
    preference_score,
    r,
    N,
    mode,
    exact=False,
    use_log=False
):
    """
    Compute the incentive for a given preference score vector and r.
    """
    if mode=='self':
        Ep=None
    else:
        Ep=np.mean(preference_score)
    if exact:
        dev=reg_prob_derivative_exact(mode,r, preference_score, T_create, R_create, num_experiments=50000, num_sampled_data=N, use_log=use_log)
    else:
        R_func=partial(R_create, Ep, mode)
        dev=reg_prob_derivative(mode, r, preference_score, T_create, R_func, num_experiments=50000, num_sampled_data=N, use_log=use_log)
    return dev


def histogram_plot(train_dataset,name=''):
    ####plot histogram
    preference_score = train_dataset['preference_score']


    plt.figure(figsize=(10, 10))  
    plt.hist(preference_score, bins=30, edgecolor='black')  # Adjust 'bins' as needed

    plt.xlabel("Score")
    plt.ylabel("Frequency")
    os.makedirs("fig", exist_ok=True)
    sv_name=name+'_'
    sv_name+='_histogram.png'
    plt.savefig("fig/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure

    plt.show()

def calibrate_data(train_dataset,eval_dataset,n_bins=20):
    preference_score = np.array(train_dataset['preference_score'])
    labels=np.array(train_dataset['label'])
    histogram = HistogramCalibrator(n_bins=n_bins)
    histogram.fit(preference_score, labels)
    histogram_probs = histogram.predict(np.array(eval_dataset['preference_score']))
    return histogram_probs
def calibrate_plot(eval_dataset,histogram_probs,n_bins=20):
    score_col = 'score'
    label_col='label'
    df_pre = pd.DataFrame({
        label_col: eval_dataset['label'],
        score_col: eval_dataset['preference_score']
    })
    df_post = pd.DataFrame({
        label_col: eval_dataset['label'],
        score_col:  histogram_probs
    })

    # key to the dictionary is for giving the result
    # a descriptive name
    eval_dict = {
        'Before Calibrate': df_pre,
        'After Calibrate': df_post
    }


    plt.rcParams['figure.figsize'] = [10, 8]
  
    df_result = compute_calibration_summary(eval_dict, label_col, score_col, n_bins=n_bins)
    print(df_result)

def incentive_plot(
    name #data can be 'sky','PKU','Helpsteer','Ultra'
):
    # change default style figure and font size
    
    plt.rcParams['font.size'] = 16
    n_bins = 30

    # Set the random seed for reproducibility
    seed = 4    
    set_random_seed(seed)

    mode='self'
    exact=True

    r_list=[0.1,0.3,0.5,0.7,0.9]
    r_list=np.linspace(0.01,0.9,101)

    N_list=[10,50,100,500,1000]



    train_dataset,eval_dataset=get_data(script_config_path='/home/zhongzec24/RewardModeling/paper_experiment_configs/llama-'+name+'.json')
    histogram_plot(eval_dataset,name=name)
    
    if name=='sky':    
        histogram_probs=calibrate_data(train_dataset,eval_dataset,n_bins=n_bins)
        calibrate_plot(eval_dataset,histogram_probs,n_bins=n_bins)
        #for elements in histogram_probs, if it is less than 0.5, set them to 1-histogram_probs
        
    else:
        histogram_probs=np.array(eval_dataset['preference_score'])
    histogram_probs=np.where(histogram_probs<0.5,1-histogram_probs,histogram_probs)

    incentive_list=[]
    for r in r_list:
        for N in N_list:
            incentive=get_incentive(histogram_probs,r,N,mode,exact)
            incentive_list.append(incentive)
    incentive_list=np.array(incentive_list).reshape(len(r_list),len(N_list))

    plt.figure(figsize=(10,10))
    for i,r in enumerate(r_list):
        plt.plot(N_list,incentive_list[i],label=f"r={r}",marker='o')
    plt.xlabel("Number of Samples")
    plt.ylabel("Incentive")
    plt.legend()
    os.makedirs("fig", exist_ok=True)
    sv_name=name+'_'+mode+'_'
    sv_name+='_incentive_n_sample.png'
    plt.savefig("fig/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure
    '''
    #plot x-axix as r
    plt.figure(figsize=(10,10))
    for i,N in enumerate(N_list):
        plt.plot(r_list,incentive_list[:,i],label=f"N={N}",marker='o')
    plt.xlabel("r")
    plt.ylabel("Incentive")
    plt.legend()
    os.makedirs("fig", exist_ok=True)
    sv_name=name+'_'+mode+'_'
    sv_name+='_incentive_r.png'
    plt.savefig("fig/"+sv_name, dpi=300,bbox_inches='tight')  # dpi=300 for high-resolution figure
    '''


def main():
    for data_name in ['sky','PKU','Helpsteer','Ultra']:
        incentive_plot(data_name)



if __name__ == "__main__":
    main()


