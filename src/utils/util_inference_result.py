
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import sys
from importlib import reload # 自訂 function 可重新 reload
sys.path.append('/home/hchuang/Documents/Project/SSSD_CP/src/datasets_script/NYISO')
from utils.data_utils import *

def read_multiple_imputations(folder_path, missing_k):
    """
    goal: read multiple imputations that are generated from 'inference_multiples.py', shape (obs, channel, length)
    output: return an array, shape (number of imputation files, obs, channel, missing_k)
    input:
        folder_path = the folder that contains the file we want to read
        missing_k = same 'missing_k' in config file, i.e, the number of the last elements to be predicted, ex: 24 in my case
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Get a list of files in the folder
    file_list = os.listdir(folder_path)

    # Filter out only imputation0.npy files
    npy_files = [file for file in file_list if file.endswith("imputation0.npy")]

    if not npy_files:
        print(f"No imputation0.npy files found in '{folder_path}'.")
        return

    # Loop through the all imputation0.npy files and read them
    stack_array_data = None
    for npy_file in npy_files:
        array_data = read_missing_k_data(folder_path, npy_file, missing_k)
        array_data = array_data.reshape(tuple([1]) + array_data.shape) # array_data.shape = (obs, channel, length) -> (1, obs, channel, length)

        if stack_array_data is None:
            stack_array_data = array_data
        else:
            stack_array_data = np.vstack((stack_array_data, array_data))

    # print(f"return (1) Load and stack all imputation0.npy files, shape: {stack_array_data.shape} = (files, obs, channels, length)")

    return stack_array_data



def read_missing_k_data(folder_path, npy_file, missing_k):
    """
    goal: return the last 24 elements of each observation
    output: an array, shape (obs, channel, length = missing_k)
    input:
        folder_path = the folder that contains the file we want to read
        npy_file = the file name that we want to read, ex: "original0.npy"
        missing_k = same'missing_k' in config file, i.e, the number of the last elements to be predicted, ex: 24 in my case
    """
    file_path = os.path.join(folder_path, npy_file)
    true = np.load(file_path)
    true = true[:,:,(-missing_k):]
    # print(f"return test data with imputation, shape{true.shape}")
    return true  
def pred_interval(pred, beta):
    """
    goal: compute the (1-alpha) quantile of imputation ecdf, i.e, prediction interval
    output: lower bound and upper bound, shape: (obs, channel, length)
    input:
        pred = all data, shape(number of imputation files, obs, channel, length)
        beta = significance level of original prediction interval
    """
    # beta = 0.05
    # compute original prediciton intervals
    L = np.quantile(pred, beta/2, axis=0)
    U = np.quantile(pred, 1-beta/2, axis=0)

    return L, U

def compute_E_star(L, U, true, alpha):
    """
    goal: compute the (1-alpha) quantile of conformity scores, i.e, E_star
    output: E_star, shape: (channel, length)
    input:
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)        
        alpha = miscoverage rate of conformal prediction
    """

    # alpha = 0.05

    # compute the conformity scores
    E = np.maximum(L-true, true-U)
    
    # compute the (1-alpha) quantile of conformity scores
    CP_PAR = (1+1/true.shape[0])*(1-alpha)
    if CP_PAR > 1:
        CP_PAR = 1
    E_star = np.quantile(E, CP_PAR, axis=0)

    return E_star

def adjust_PI(L, U, E_star):
    """
    goal: adjust prediction interval using conformal prediction
    output: adjusted lower and upper bound, shape: (obs, channel, length)
    input: 
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)
        E_star = scores, shape: (channel, length)
    """
    E_star_exd = np.expand_dims(E_star, axis=0)
    return L-E_star_exd, U+E_star_exd    

def compute_E_star_separate(L, U, true, alpha):
    """
    goal: compute the (1-alpha) quantile of conformity scores for lower and upper bound, respectively, i.e, E_star
    output: E_star_L and E_star_U, shape: (channel, length)
    input:
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)
        alpha = miscoverage rate of conformal prediction
    """

    # alpha = 0.05

    # lower bound
    ## compute the conformity scores
    E = L-true
    ## compute the (1-alpha) quantile of conformity scores
    CP_PAR = (1+1/true.shape[0])*(1-alpha)
    E_star_L = np.quantile(E, CP_PAR, axis=0)

    # upper bound
    ## compute the conformity scores
    E = true - U
    ## compute the (1-alpha) quantile of conformity scores
    CP_PAR = (1+1/true.shape[0])*(1-alpha)
    E_star_U = np.quantile(E, CP_PAR, axis=0)

    return E_star_L, E_star_U

def adjust_PI_separate(L, U, E_star_L, E_star_U):
    """
    goal: adjust prediction interval using compute_E_star_separate
    output: adjusted lower and upper bound, shape: (obs, channel, length)
    input: 
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)
        E_star = scores, shape: (channel, length)
    """
    E_star_L_exd = np.expand_dims(E_star_L, axis=0)
    E_star_U_exd = np.expand_dims(E_star_U, axis=0)
    return L-E_star_L_exd, U+E_star_U_exd

def compute_E_star_loadp(L, U, true, alpha, X):
    """
    loadp = locally adaptive
    goal: compute the (1-alpha) quantile of conformity scores, i.e, E_star
    output: E_star, shape: (channel, length)
    input:
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)        
        alpha = miscoverage rate of conformal prediction
        X = independent variable(obs, channel, length of given data = 168)
    """

    sd = np.std(X, axis = 2)
    sd = sd.reshape(X.shape[0], X.shape[1], 1)

    # compute the conformity scores
    E = np.maximum((L-true)/sd, (true-U)/sd)
    
    # compute the (1-alpha) quantile of conformity scores
    CP_PAR = (1+1/true.shape[0])*(1-alpha)
    if CP_PAR > 1:
        CP_PAR == 0.99
    E_star = np.quantile(E, CP_PAR, axis=0)

    return E_star

def adjust_PI_loadp(L, U, E_star, X):
    """
    loadp = locally adaptive
    goal: adjust prediction interval using conformal prediction
    output: adjusted lower and upper bound, shape: (obs, channel, length)
    input: 
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)
        E_star = scores, shape: (channel, length)
        X = independent variable(obs, channel, length of given data = 168)
    """

    sd = np.std(X, axis = 2)
    sd = sd.reshape(X.shape[0], X.shape[1], 1)

    E_star_exd = np.expand_dims(E_star, axis=0)
    return L-E_star_exd*sd, U+E_star_exd*sd

def coverage_rate(L, U, true):
    """
    goal: compute the coverage rate, which is the proportion of [L,U] contains true data 
    output: an list containing array(s), array shape = (24,). # of list = # of channels of input data
    input:
        L = lower bound, shape: (2209, 1, 24)
        U = upper bound, shape: (2209, 1, 24)
        true = true data, shape: (2209, 1, 24)
    """
    return np.sum(np.logical_and(true > L, true < U), axis=0)/true.shape[0]

def generate_date_from_seq(value):
    """
    goal: what is the date of given number of obs
    output: date
    input: a value (obs)
    """
    # start_date = datetime.strptime("2019-01-08", "%Y-%m-%d")
    start_date = datetime.strptime("2016-10-20", "%Y-%m-%d")
    days_offset = value
    target_date = start_date + timedelta(days=days_offset)
    formatted_date = target_date.strftime("%Y/%m/%d")
    
    return formatted_date


def compute_E_star_SCP(pred, true, alpha = 0.05):
    """
    goal: split conformal prediction
    output: lower and upper bounds of conformal prediciton
    input: 
        pred = prediction, shape: (obs, channel, length)
        true = true data, shape: (obs, channel, length)
        alpha = miscoverage rate of conformal prediction
    """

    ## compute the conformity scores
    E = np.abs(pred-true)
    ## compute the (1-alpha) quantile of conformity scores
    CP_PAR = (1+1/true.shape[0])*(1-alpha)
    E_star = np.quantile(E, CP_PAR, axis=0)
    
    ## reshape (1,24) to (1,1,24)
    return np.expand_dims(E_star, axis=0) 



def CR_barchart_compare(original_values, adjusted_values2, 
                        figure_title = "Comparing 95% PI Coverage Rate: SSSD vs. Calibration",
                        label_1 = 'SSSD', label_2 = 'Calibration',
                        color_1 = 'tab:orange', color_2 = 'tab:blue'):
    """
    original_values: array, shape = (24,)
    adjusted_values2: array, shape = (24,)
    """
    X_axis = np.arange(original_values.shape[0]) 

    plt.figure(figsize=(20, 10)) 
    plt.rcParams['font.size'] = 24 # 設置繪圖時的字體大小
    plt.bar(X_axis - 0.2, original_values, 0.4, label = label_1, color = color_1) 
    plt.bar(X_axis + 0.2, adjusted_values2, 0.4, label = label_2, color = color_2) 

    # for i in X_axis:
        # plt.text(X_axis[i] - 0.2, original_values[i] + 0.02, str(round(original_values[i],2)), ha='center', va='bottom')
        # plt.text(X_axis[i] + 0.2, adjusted_values2[i] + 0.03, str(round(adjusted_values2[i],2)), ha='center', va='bottom')

    plt.axhline(y=0.95, color='r', linestyle='--', label='0.95')
    plt.axhline(y=1, color='black', linestyle='--', label='0.95')

    plt.ylim(0, 1.1)
    plt.xlabel("Hours") 
    plt.ylabel("") 
    plt.title(figure_title) 
    # plt.legend(loc = 'upper right', bbox_to_anchor=(1, 0.5)) 
    plt.legend(loc=(1.001, 0.79))
    plt.show() 

def err_boxplot(error_data, 
                figure_title = 'Boxplot of the absolute error'):
    """
    input:
        error_data is an array, shape: (# of obs, channel = 1, # of length)
    output: boxplot with  # of boxes = # of length, # of points of each boxes = # of obs
    """

    data_to_plot = np.squeeze(abs_err_I2) # reduce the channel dimension
    box_data = data_to_plot.T.tolist()
    # box_data is a list containing lists. 
    # The number of elements in the 1st level list represents the number of boxes.
    # The number  of elements in the 2nd level list represent the number of points of one box

    plt.figure(figsize=(10, 6))
    plt.boxplot(box_data) 
    plt.xlabel("Hours")
    plt.ylabel("Values")
    plt.title("Boxplot of the absolute error of the I2 dataset")
    plt.grid(True)
    plt.ylim((0,2))
    plt.show()

def prediction_linechart(obs, y, y_predict):
    """
    input: 
        one_obs: specify a number from the size of data (ie specify which series should plot)
        y: data values, an array, shape = (obs, length = 192)
        y_predict: prediciton, an array, shape = (obs, length = 24)
    """
    date = generate_date_from_seq(obs)
    
    x = range(0,y.shape[1])
    x_predict = range(y.shape[1]-y_predict.shape[1], y.shape[1])
    
    y_single = y[obs,:]
    y_predict_single = y_predict[obs,:]

    plt.figure(figsize=(16, 4)) 
    plt.plot(x, y_single, color = 'tab:gray', label = "Data")
    plt.plot(x_predict, y_predict_single, color = 'tab:blue', label = "Prediction")
    plt.title("" + date)
    # plt.xticks(rotation=45) 
    plt.ylim(-2.5, 2.5) 
    plt.legend(loc = "upper left")
    plt.show()

def PI_linechart(obs,y, L, U, figure_title = "Prediction interval", PI_label = "PI", PI_color = 'tab:blue'):
    """
    input: 
        one_obs: specify a number from the size of data (ie specify which series should plot)
        y: data values, an array, shape = (obs,  length = 192)
        L: lower bound of PI, an array, shape = (obs, length = 24)
        U: upper bound of PI, an array, shape = (obs, length = 24)
    """    
    # date = generate_date_from_seq(obs)
    
    x = range(0,y.shape[1])
    x_predict = range(y.shape[1]-L.shape[1], y.shape[1])

    y_single = y[obs,:]
    lower = L[obs,:]
    upper = U[obs,:]

    plt.figure(figsize=(16, 4)) 
    plt.rcParams['font.size'] = 20 # 設置繪圖時的字體大小

    plt.plot(x, y_single, color = 'tab:gray', label = "Data")
    # plt.title(figure_title +' ('+date+')')

    # plt.xticks(rotation=45) 
    plt.fill_between(x_predict, lower, upper, color=PI_color, alpha=0.4, label = PI_label)
    # plt.ylim(-2.5, 2.5) 
    plt.legend(loc = "upper left")
    plt.show()    
def indicator_function(condition, shape):
    # Perform element-wise comparison and return 1 if condition is met, 0 otherwise
    return np.where(condition, 1, 0)
def Interval_Score(l,u,z,alpha):
    return (u-l)+2/alpha*(l-z)*indicator_function(z<l,z.shape)+2/alpha*(z-u)*indicator_function(z>u,z.shape)    

def load_result_data(main_folder_path, DATASET):
    if DATASET == 'dataset_4': # (test data = 73)
        result_folder_path = "results/NYISO_4/NYISO_4_NYC_split/T200_beta00.0001_betaT0.02/"
        I2_foldername = "imputaiton_multiple_20k_I2_part"
        test_foldername = "imputaiton_multiple_20k_test/"

        I2_dataset_folder_path = 'datasets/NYISO/dataset_4/zone_split/'
        I2_dataset_filename = 'I2_N.Y.C._train.npy'
        test_dataset_folder_path = 'datasets/NYISO/dataset_4/zone/'
        test_dataset_filename = 'N.Y.C._test.npy'
    else: # (test data = 350)
        
        result_folder_path = "results/NYISO_4/NYISO_4_NYC_split/T200_beta00.0001_betaT0.02/"
        I2_foldername = "imputaiton_multiple_20k_I2_part"
        test_foldername = 'imputaiton_multiple_20k_test_dataset6'

        I2_dataset_folder_path = 'datasets/NYISO/dataset_6/zone_split/'
        I2_dataset_filename = 'I2_N.Y.C._train.npy'
        test_dataset_folder_path = 'datasets/NYISO/dataset_6/zone/'
        test_dataset_filename = 'N.Y.C._test.npy'


    # I2 data in original scael
    file_path = os.path.join(main_folder_path + I2_dataset_folder_path, I2_dataset_filename)
    true_data_I2_all_o = np.squeeze(np.swapaxes(np.load(file_path), axis1=1, axis2=2))
    true_data_I2_cond_o = true_data_I2_all_o[:,0:168]
    true_data_I2_target_o = true_data_I2_all_o[:,-24:]
    # print(f"true_data_I2_all_o, shape: {true_data_I2_all_o.shape}")
    # print(f"true_data_I2_cond_o, shape: {true_data_I2_cond_o.shape}")
    # print(f"true_data_I2_target_o, shape: {true_data_I2_target_o.shape}")

    # I2 data in normalized scale
    m = np.mean(true_data_I2_cond_o, axis=1).reshape(1, true_data_I2_cond_o.shape[0], 1)
    s = np.std(true_data_I2_cond_o, axis=1).reshape(1, true_data_I2_cond_o.shape[0], 1)
    true_data_I2_all =  np.squeeze((true_data_I2_all_o-m)/s)
    true_data_I2_cond = np.squeeze((true_data_I2_cond_o-m)/s)
    true_data_I2_target = np.squeeze((true_data_I2_target_o-m)/s)
    print(f"true_data_I2_all, shape: {true_data_I2_all.shape}")
    print(f"true_data_I2_cond, shape: {true_data_I2_cond.shape}")
    print(f"true_data_I2_target, shape: {true_data_I2_target.shape}")


    # I2 prediction data in normalized scael
    mul_pred_data_I2 = np.squeeze(np.vstack([read_multiple_imputations(main_folder_path + result_folder_path + I2_foldername + str(i) +'/', 24) for i in range(3)]))
    median_pred_data_I2 = np.median(mul_pred_data_I2, axis = 0)
    print(f"mul_pred_data_I2, shape: {mul_pred_data_I2.shape}")
    print(f"median_pred_data_I2, shape: {median_pred_data_I2.shape}")


    # pred I2 data back to original scale
    mul_pred_data_I2_o = mul_pred_data_I2*s + m
    median_pred_data_I2_o = np.squeeze(np.median(mul_pred_data_I2_o, axis = 0))
    # print(f"mul_pred_data_I2_o, shape: {mul_pred_data_I2_o.shape}")
    # print(f"median_pred_data_I2_o, shape: {median_pred_data_I2_o.shape}")

    # test data in original scale
    file_path = os.path.join(main_folder_path + test_dataset_folder_path, test_dataset_filename)
    true_data_test_all_o = np.squeeze(np.swapaxes(np.load(file_path), axis1=1, axis2=2))
    true_data_test_cond_o = true_data_test_all_o[:,0:168]
    true_data_test_target_o = true_data_test_all_o[:,-24:]

    # print(f"true_data_test_all_o, shape: {true_data_test_all_o.shape}")
    # print(f"true_data_test_cond_o, shape: {true_data_test_cond_o.shape}")
    # print(f"true_data_test_target_o, shape: {true_data_test_target_o.shape}")

    # test data in normalized scale
    m = np.mean(true_data_test_cond_o, axis=1).reshape(1, true_data_test_cond_o.shape[0], 1)
    s = np.std(true_data_test_cond_o, axis=1).reshape(1, true_data_test_cond_o.shape[0], 1)
    true_data_test_all =  np.squeeze((true_data_test_all_o-m)/s)
    true_data_test_cond =  np.squeeze((true_data_test_cond_o-m)/s)
    true_data_test_target =  np.squeeze((true_data_test_target_o-m)/s)
    print(f"true_data_test_all, shape: {true_data_test_all.shape}")
    print(f"true_data_test_cond, shape: {true_data_test_cond.shape}")
    print(f"true_data_test_target, shape: {true_data_test_target.shape}")

    # pred test dataset in normalized scale
    mul_pred_data_test = np.squeeze(read_multiple_imputations(main_folder_path + result_folder_path + test_foldername, 24))
    median_pred_data_test = np.median(mul_pred_data_test, axis = 0)
    print(f"mul_pred_data_test, shape: {mul_pred_data_test.shape}")
    print(f"median_pred_data_test, shape: {median_pred_data_test.shape}")

    # pred test data back to original scale
    mul_pred_data_test_o = mul_pred_data_test*s + m
    median_pred_data_test_o = np.median(mul_pred_data_test_o, axis = 0)
    # print(f"mul_pred_data_test_o, shape: {mul_pred_data_test_o.shape}")
    # print(f"median_pred_data_test_o, shape: {median_pred_data_test_o.shape}")
    return {
        # I2 data - normalized scale
        'true_data_I2_all': true_data_I2_all,
        'true_data_I2_cond': true_data_I2_cond,
        'true_data_I2_target': true_data_I2_target,
        
        # I2 data - original scale
        'true_data_I2_all_o': true_data_I2_all_o,
        'true_data_I2_cond_o': true_data_I2_cond_o,
        'true_data_I2_target_o': true_data_I2_target_o,

        # I2 prediction - normalized & original
        'mul_pred_data_I2': mul_pred_data_I2,
        'median_pred_data_I2': median_pred_data_I2,
        'mul_pred_data_I2_o': mul_pred_data_I2_o,
        'median_pred_data_I2_o': median_pred_data_I2_o,

        # test data - normalized scale
        'true_data_test_all': true_data_test_all,
        'true_data_test_cond': true_data_test_cond,
        'true_data_test_target': true_data_test_target,
        
        # test data - original scale
        'true_data_test_all_o': true_data_test_all_o,
        'true_data_test_cond_o': true_data_test_cond_o,
        'true_data_test_target_o': true_data_test_target_o,

        # test prediction - normalized & original
        'mul_pred_data_test': mul_pred_data_test,
        'median_pred_data_test': median_pred_data_test,
        'mul_pred_data_test_o': mul_pred_data_test_o,
        'median_pred_data_test_o': median_pred_data_test_o
    }


def date_index(zone_number = 1):
    # 0-read file
    df_all_time = pd.read_pickle('/home/hchuang/Documents/Project/SSSD_CP/src/datasets/NYISO/pickle/df_all_time.pkl')
    # 1-the day of the year
    date_range = pd.date_range(start='2005-01-31', end='2016-10-19')
    days = date_range.to_series().dt.dayofyear # 創建一個 Series，包含每個日期是一年當中第幾天
    train_all_days = days[7:]

    # 2-training data (在切割training and test data時會drop有nan的series，所以需要知道其 index)
    df = df_all_time
    train_start = '2005-01-31 00:00:00'
    train_end = '2016-10-19 23:00:00'
    test_start = '2016-10-13 00:00:00'
    test_end = '2016-12-31 23:00:00'
    zone_number = 1
    days_window = 8
    zone_name = 'N.Y.C.'

    if zone_name == 'ALL':
        df_train = df[(df['Date']>=train_start) 
                    & (df['Date']<=train_end)] 
        df_test  = df[(df['Date']>=test_start) 
                    & (df['Date']<=test_end)]
    else:
        df_train = df[(df['Zone']==zone_name) 
                    & (df['Date']>=train_start) 
                    & (df['Date']<=train_end)] 
        df_test  = df[(df['Zone']==zone_name) 
                    & (df['Date']>=test_start)
                    & (df['Date']<=test_end)]
        
    np_train, day_of_week_train = pd_to_numpy(df_train, zone_number, days_window) # np_train.shape = 4273
    # remove rows (reduce obs.) which contain nan,  (obs., channel, length)
    load_array_train = np_train[~np.isnan(np_train).any(axis=(1,2))] 
    # exchange channel and length (obs., 1, length) -> (obs., length, 1) 
    load_array_train = np.einsum('ijk->ikj',load_array_train)


    # 3-根據train沒有因為na被drop的index選取days
    train_days = train_all_days[~np.isnan(np_train).any(axis=(1,2))] 

    # 4-根據I1和I2 index切分 dyas
    data_npy = load_array_train
    np.random.seed(42)

    random_indices = np.random.choice(data_npy.shape[0], data_npy.shape[0]//2, replace=False)
    remaining_indices = np.setdiff1d(np.arange(data_npy.shape[0]), random_indices)

    sample_I1 = data_npy[random_indices]
    sample_I2 = data_npy[remaining_indices]

    train_days_I1 = train_days.iloc[random_indices]
    train_days_I2 = train_days.iloc[remaining_indices]

    return train_days_I1, train_days_I2