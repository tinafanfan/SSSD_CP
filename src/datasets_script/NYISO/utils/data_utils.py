import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def printsth(sth):
    # print(sth)
    print("tina")

def merge_all_time(df):
    '''
    補上所有時間點，創造應該補值的row
    df: 包含 'Date', 'Zone', 'Load' 的 dataframe

    output: a data frame with same columns. The number of rows is hours_df.shape[0]*11
    '''
    # 創造所有的 hour 時間點
    hours_df = pd.DataFrame({'Date': pd.date_range(start = df['Date'].min(), end = df['Date'].max(), freq = '1H') })

    zones = df['Zone'].unique()

    result_all_time = pd.DataFrame()
    for zone in zones:

        # one zone in one loop
        load_zone = df.loc[df['Zone'] == zone,]

        # merge to hourly hours
        result = pd.merge(hours_df, load_zone, on='Date', how='left')
        result['Zone'] = zone

        result_all_time = pd.concat([result_all_time, result], 
                                    axis=0)
    return result_all_time

def vis_na(result, figsize_width, figsize_height):
    '''
    present a heatmap of NA, rows are days and columns are hours of each date
    result: a dataframe which the number of rows is multiples of 24 (24的倍數)
    figsize_width: width of figsize
    figsize_height: height of figsize
    '''
    result = result.reset_index() # result 的 index 不一定是從 0 開始，所以 reset，使 result.loc[0,'Date'] 可以使用
    no_day = int(result.shape[0]/24)

    # create a day list
    start_date = pd.to_datetime(result.loc[0,'Date'], format='%Y%m%d') # 将初始日期转换为 Pandas 的 datetime 格式
    date_series = pd.date_range(start_date, periods = no_day, freq='D') # 创建一个日期序列
    date_list = date_series.strftime('%Y%m%d') # 将日期序列转换为字符串格式

    # reshape data
    NA = result['Load'].isna()
    NA_vec = NA.to_numpy()
    heatmap_data = NA_vec.reshape(no_day, 24) # 將向量重新排列為.x24的形狀
    
    # set heat map color: 黑色 FALSE，灰色 TRUE (NA)
    custom_cmap = plt.matplotlib.colors.ListedColormap(['black', 'gray']) 

    # figure setting
    plt.figure(figsize=(figsize_width, figsize_height))  # 设置图的宽度和高度
    plt.yticks(range(len(date_list)), date_list) # 在每一行的左侧显示行号
    plt.imshow(heatmap_data, cmap=custom_cmap, aspect='auto') # 绘制热图
    plt.show() # 显示热图

def vis_na_specify(result, start, end, figsize_width, figsize_height):
    
    '''
    present a similar heatmap of vis_na, the different is vis_na_specify can specify a start and an end time
    start: like '2006-02-06 00:00:00'
    end: like '2006-02-07 00:00:00'
    figsize_width: width of figsize
    figsize_height: height of figsize    
    '''
    a = result.loc[result['Date'] == start].index.values[0]
    b = result.loc[result['Date'] == end].index.values[0] + 1
    vis_na(result[a:b], figsize_width, figsize_height)

    # .index 取出來會是 Int64Index，要用 .values[0] 將其中的數值取出


def count_year_na(df_one_zone):
    # 計算每年遺失值個數
    df_one_zone.set_index('Date', inplace=True)
    # daily_na_count = df['Load'].isna().resample('D').sum()
    # month_na_count = df['Load'].isna().resample('M').sum()
    year_na_count = df_one_zone['Load'].isna().resample('Y').sum()
    return year_na_count



def impute_linear(df):
    df_impute_linear = df.copy()

    df_impute_linear = df_impute_linear.set_index('Date') # set index 是為了畫圖
    imputed_indices = df_impute_linear[df_impute_linear['Load'].isna()].index

    df_impute_linear['Load'] = df_impute_linear['Load'].interpolate(method='linear')

    return df_impute_linear, imputed_indices

def impute_stl(df):
    # STL imputation

    # Make a copy of the original dataframe
    df_impute_stl = df.copy()
    df_impute_stl = df_impute_stl.set_index('Date')

    # Fill missing values in the time series
    imputed_indices = df_impute_stl[df_impute_stl['Load'].isna()].index
    # Apply STL decompostion
    stl = STL(df_impute_stl['Load'].interpolate(), seasonal = 8639)
    res = stl.fit()

    # Extract the seasonal and trend components
    seasonal_component = res.seasonal

    # Create the deseasonalised series
    df_deseasonalised = df_impute_stl['Load'] - seasonal_component

    # Interpolate missing values in the deseasonalised series
    df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear")

    # Add the seasonal component back to create the final imputed series
    df_imputed = df_deseasonalised_imputed + seasonal_component

    # Update the original dataframe with the imputed values
    df_impute_stl.loc[imputed_indices, 'Load'] = df_imputed[imputed_indices]

    return df_impute_stl, imputed_indices


def plot_impute_result(df_impute_stl, imputed_indices):
    # Plot the series using pandas
    plt.figure(figsize=[20, 6])
    df_impute_stl['Load'].plot(style='.-',  label='Load')
    plt.scatter(imputed_indices, df_impute_stl.loc[imputed_indices, 'Load'], color='red')

    plt.title("Sales with STL Imputation")
    plt.ylabel("Sales")
    plt.xlabel("Time")
    plt.show()


def count_zone_na(df):
    '''
    呈現每個區域總NA值和分年NA值
    '''
    zones = df['Zone'].unique()

    total_na = []
    month_na = []
    for zone in zones:

        # one zone in one loop
        load_zone = df.loc[df['Zone'] == zone,]
        # Total NA
        total_na.append(load_zone['Load'].isna().sum())
        # NA by year
        month_na.append(count_year_na(load_zone))
    
    return total_na, month_na


def pd_to_numpy(load, zones_num, days_window):
    '''
    將一個 dataframe 轉成 np array (observation, channel, length)

    此處以 days_window 天作為一個時間序列
    共有 6433 個 observation，11個 zone作為 channel，days_window x 24 為 length

    input: 
        load: dataframe with columns: Date, Load, Zone
        zones_num: how many zones(channels)
    output: np array (observation, channel, length)
    '''

    # 使用 pd.date_range 生成日期序列
    days = pd.date_range(start = load['Date'].min(), 
                         end = load['Date'].max() - pd.Timedelta(days=(days_window-1)), 
                         freq = '1D') 

    load_array = np.empty((1, zones_num, days_window*24 ))


    # 產生 np array (6433, 11, days_window x 24)
    dim = (1, zones_num, days_window*24)
    load_array = np.empty(dim) # (1,11, days_window x 24 )
    for day in days:
        # 挑選 days_window 天資料
        start_date = pd.to_datetime(day)
        
        end_date = start_date + pd.Timedelta(days = days_window)
        
        selected_data = load.loc[(load['Date'] >= start_date) & (load['Date'] < end_date),:]
        
        # df -> numpy array [obs(number of loops) = 6433, channel(number of zones) = 11, length(number of hours) = days_window x 24]
        pivot_df = selected_data.pivot(index='Zone', columns='Date', values='Load') 
        pivot_np = pivot_df.to_numpy() # (zone, Date)
        
        pivot_np = pivot_np.reshape(dim) # (1,11,days_window x 24)
        load_array = np.concatenate((load_array, pivot_np), axis=0)

    return load_array[1:] # delete the 1 st slice created by np.empty(dim)


def train_test_select(data_ls, train_start, train_end, test_start, test_end, zone_number, days_window, zone_name='ALL'):  

    '''
    根據時間劃分，將 npy 分成 training set 和 testing set
    可以取單一 zone (zone_name = 'N.Y.C.') 或所有 zone (zone_name = 'ALL')

    input: data_ls 長度為 8 (8 periods)的 list，每個元素都是 dataframe
    output: 兩個 npy (obs, length, channel)
    '''
    np_list_train = []
    np_list_test = []
    for df in data_ls:
        
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
        
        # 沒有選到的 period (ex:2020) shape[0]就會是0，會有 error
        if df_train.shape[0] > 0:
            np_list_train.append(pd_to_numpy(df_train, zone_number, days_window))
        if df_test.shape[0] > 0:
            np_list_test.append(pd_to_numpy(df_test, zone_number, days_window))

    load_array_train = np.vstack(np_list_train)
    load_array_test = np.vstack(np_list_test)

    # exchange channel and length (obs., 1, length) -> (obs., length, 1) 
    load_array_train = np.einsum('ijk->ikj',load_array_train)
    load_array_test = np.einsum('ijk->ikj',load_array_test)

    return load_array_train, load_array_test


def z_normalization(data, days_normalized):
    '''
    將每一條 ts 做 normalization (x-mean)/sd
    
    input:  npy (obs, length, channel)
    output: npy (obs, length, channel)
    '''

    obs = data.shape[0]
    channel = data.shape[2]

    mean = np.mean(data[:, 0:(days_normalized*24), :], axis = 1).reshape(obs, 1, channel)
    std  = np.std(data[:, 0:(days_normalized*24), :], axis = 1).reshape(obs, 1, channel)
    data_normalized = (data - mean)/std

    return data_normalized

def range_normalization(data, days_normalized):
    '''
    將每一條 ts 做 normalization (x-min)/(max - min)
    
    input:  npy (obs, length, channel)
    output: npy (obs, length, channel)
    '''

    obs = data.shape[0]
    channel = data.shape[2]

    min_value = np.min(data[:, 0:(days_normalized*24), :], axis = 1).reshape(obs, 1, channel)
    max_value = np.max(data[:, 0:(days_normalized*24), :], axis = 1).reshape(obs, 1, channel)
    
    data_normalized = (data - min_value)/(max_value - min_value)

    return data_normalized



def plot_two_lines(x1, y1, x2, y2, lagend1, legend2):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 4))

    # Plot the line
    ax.plot(x1, y1, color = '#1f77b4') # blue
    ax.plot(x2, y2, color = '#ff7f0e') # orange

    # Set labels and title
    plt.title('NYISO forecast')
    ax.legend([lagend1, legend2])
    ax.set(xlabel=None, ylabel='value', title=str(''))

    # Rotate x-axis tick labels for better visibility
    plt.xticks(rotation=45)
    # Display the plot   

    # fig.savefig('.png'.format(x))
    fig.show()
