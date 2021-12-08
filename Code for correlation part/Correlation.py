import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

# Social Signal
df_sentiment = pd.read_csv('./data_for_correlation/twitter_sentiment.csv')
df_emotion = pd.read_csv('./data_for_correlation/twitter_emotion.csv')
df_suicide = pd.read_csv('./data_for_correlation/twitter_suicide.csv')

df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
df_sentiment.set_index(keys='date', inplace=True)
df_emotion['date'] = pd.to_datetime(df_emotion['date'])
df_emotion.set_index(keys='date', inplace=True)
df_suicide['date'] = pd.to_datetime(df_suicide['date'])
df_suicide.set_index(keys='date', inplace=True)

# Normalize
df_month = pd.read_csv('./data_for_correlation/month.csv')

normalize = True

if normalize:
    df_sentiment['positive'] = df_sentiment.positive.values / df_month.comments.values
    df_sentiment['negative'] = df_sentiment.negative.values / df_month.comments.values
    df_sentiment['ambiguous'] = df_sentiment.ambiguous.values / df_month.comments.values
    df_sentiment['neutral'] = df_sentiment.neutral.values / df_month.comments.values
    
    df_emotion['anger'] = df_emotion.anger.values / df_month.comments.values
    df_emotion['disgust'] = df_emotion.disgust.values / df_month.comments.values
    df_emotion['fear'] = df_emotion.fear.values / df_month.comments.values
    df_emotion['joy'] = df_emotion.joy.values / df_month.comments.values
    df_emotion['sadness'] = df_emotion.sadness.values / df_month.comments.values
    df_emotion['surprise'] = df_emotion.surprise.values / df_month.comments.values
    
    df_suicide['normal'] = df_suicide.normal.values / df_month.comments.values
    df_suicide['suicidal'] = df_suicide.suicidal.values / df_month.comments.values

# Sentiment
df_positive = df_sentiment['positive']
df_negative = df_sentiment['negative']
df_ambiguous = df_sentiment['ambiguous']
df_neutral = df_sentiment['neutral']

# Emotion
df_anger = df_emotion['anger']
df_disgust = df_emotion['disgust']
df_fear = df_emotion['fear']
df_joy = df_emotion['joy']
df_sadness = df_emotion['sadness']
df_surprise = df_emotion['surprise']

# Suicide
df_normal = df_suicide['normal']
df_suicidal = df_suicide['suicidal']

social_signal = {
    'positive': df_positive,
    'negative': df_negative,
    'ambiguous': df_ambiguous,
    'neutral': df_neutral,
    'anger': df_anger,
    'disgust': df_disgust,
    'fear': df_fear,
    'joy': df_joy,
    'sadness': df_sadness,
    'surprise': df_surprise,
    'normal': df_normal,
    'suicidal': df_suicidal
}

# Ground-truth
df_groundtruth_1 = pd.read_csv('./data_for_correlation/depression_diagnosed_patients.csv')
df_groundtruth_2 = pd.read_csv('./data_for_correlation/selfharm.csv')

df_groundtruth_1['date'] = pd.to_datetime(df_groundtruth_1['date'])
df_groundtruth_1.set_index(keys='date', inplace=True)
df_groundtruth_2['date'] = pd.to_datetime(df_groundtruth_2['date'])
df_groundtruth_2.set_index(keys='date', inplace=True)

# Normalize
df_pop = pd.read_csv('./data_for_correlation/pop.csv')

normalize = True

if normalize:
    df_groundtruth_1['depression_diagnosed_patients'] = df_groundtruth_1.depression_diagnosed_patients.values / df_pop.Population.values

# Ground-truth 1
df_depression_diagnosed_patients = df_groundtruth_1['depression_diagnosed_patients']

# Ground-truth 2
df_home_visits = df_groundtruth_2['home_visits']
df_dead = df_groundtruth_2['dead']
df_not_dead = df_groundtruth_2['not_dead']

groundtruth = {
    'depression_diagnosed_patients': df_depression_diagnosed_patients,
    'home_visits': df_home_visits,
    'dead': df_dead,
    'not_dead': df_not_dead,
}

starting_month = pd.date_range(start='2019-01', end='2021-01', freq='M')
window_size = [6, 7, 8, 9, 10, 11, 12]
horizon = [-3, -2, -1, 0, 1, 2, 3]

def check_enough_data(df, m, w):
    df_month = df.loc[m:]
    if len(df_month) >= w:
        result = True
    else:
        result = False
    return result

def dataframe_preprocessing(df, m, w):
    df_final = df.loc[m:]
    result = df_final.iloc[:w]
    return result

results = list()

for s in social_signal:
    for g in groundtruth:
        for m in starting_month:
            for w in window_size:
                if check_enough_data(social_signal[s], m, w) & check_enough_data(groundtruth[g], m, w):
                    for h in horizon:
                        if check_enough_data(social_signal[s], m, w) & check_enough_data(groundtruth[g], m + relativedelta(months=+h), w):
                            social_series = dataframe_preprocessing(social_signal[s], m, w).values
                            groundtruth_series = dataframe_preprocessing(groundtruth[g], m + relativedelta(months=+h), w).values
                            corr = np.corrcoef(social_series, groundtruth_series)
                            data_dict = {
                                'social_signal': s,
                                'groundtruth': g,
                                'starting_month': m,
                                'window_size': w,
                                'horizon': h,
                                'correlation': corr[0][1]
                            }
                            results.append(data_dict) 

with open('./data_for_correlation/corr_result_allnormalized_dicts_final.pickle', 'wb') as result:
    pickle.dump(results, result)

df_results = pd.DataFrame(results)
df_results.to_csv('./data_for_correlation/correlation_results.csv', index=False)

# Baseline (Google Trends)
df_gt_stress = pd.read_csv('./data_for_correlation/googletrends_stress.csv')
df_gt_depression = pd.read_csv('./data_for_correlation/googletrends_depression.csv')
df_gt_suicide = pd.read_csv('./data_for_correlation/googletrends_suicide.csv')
df_gt_burnout = pd.read_csv('./data_for_correlation/googletrends_burnout.csv')

df_gt_stress['date'] = pd.to_datetime(df_gt_stress['date'])
df_gt_stress.set_index(keys='date', inplace=True)
df_gt_depression['date'] = pd.to_datetime(df_gt_depression['date'])
df_gt_depression.set_index(keys='date', inplace=True)
df_gt_suicide['date'] = pd.to_datetime(df_gt_suicide['date'])
df_gt_suicide.set_index(keys='date', inplace=True)
df_gt_burnout['date'] = pd.to_datetime(df_gt_burnout['date'])
df_gt_burnout.set_index(keys='date', inplace=True)

google_trends = {
    'googletrends_stress': df_gt_stress,
    'googletrends_depression': df_gt_depression,
    'googletrends_suicide': df_gt_suicide,
    'googletrends_burnout': df_gt_burnout,
}

starting_month = pd.date_range(start='2019-01', end='2021-01', freq='M')
window_size = [6, 7, 8, 9, 10, 11, 12]
horizon = [-3, -2, -1, 0, 1, 2, 3]

results_2 = list()

for s in google_trends:
    for g in groundtruth:
        for m in starting_month:
            for w in window_size:
                if check_enough_data(google_trends[s], m, w) & check_enough_data(groundtruth[g], m, w):
                    for h in horizon:
                        if check_enough_data(google_trends[s], m, w) & check_enough_data(groundtruth[g], m + relativedelta(months=+h), w):
                            social_series = dataframe_preprocessing(google_trends[s], m, w).values.reshape(1, -1)[0]
                            groundtruth_series = dataframe_preprocessing(groundtruth[g], m + relativedelta(months=+h), w).values
                            corr = np.corrcoef(social_series, groundtruth_series)
                            data_dict = {
                                'social_signal': s,
                                'groundtruth': g,
                                'starting_month': m,
                                'window_size': w,
                                'horizon': h,
                                'correlation': corr[0][1]
                            }
                            results_2.append(data_dict)

with open('./data_for_correlation/corr_result_googletrends_allnormalized_dicts_final.pickle', 'wb') as result:
    pickle.dump(results_2, result)

df_results = pd.DataFrame(results_2)
df_results.to_csv('./data_for_correlation/correlation_results_googletrends.csv', index=False)
