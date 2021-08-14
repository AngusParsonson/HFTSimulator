import seaborn as sns
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import statistics
import os
import math
from seaborn.categorical import boxplot

sns.set(rc={'figure.figsize':(8,5)})
# sns.set(rc={'figure.figsize':(10,5)})

lstm_100_profits = [[-112.06704545467483, 220, 1, 6.290093507491938], 
                    [433.1900454545339, 135, 0, 7.073202497082475],
                    [641.0399090909286, 406, 116, 10.529358758551108],
                    [343.52649999983623, 536, 109, 11.105914101103323],
                    [1226.5301818182452, 533, 300, 10.69711582470467],
                    [748.3365909091863, 564, 298, 11.409150485250498],
                    [458.64831818176117, 175, 0, 11.50373790829567],
                    [1558.6927272727116, 285, 0, 12.63300275824837]]

lstm_50_profits = [[-413.0096818181687, 125, 43, 7.020346755094457],
                     [130.6629999999841, 92, 2, 5.8108006463669115],
                     [328.0784090909001, 257, 191, 7.548842070687484],
                     [276.61745454547145, 325, 259, 8.492137602998579],
                     [1007.8400909090051, 325, 446, 7.887245824995818],
                     [253.69418181813853, 352, 460, 7.717048579396459],
                     [193.55263636363725, 102, 28, 8.761686597901178],
                     [420.2427272727282, 235, 2, 10.30309084742008]]

lstm_20_profits = [[75.84390909087779, 561, 33, 4.5168552319021575],
                   [260.63627272726717, 423, 13, 3.5728876493631865],
                   [644.7985454545096, 783, 173, 4.373731980965075],
                   [331.70427272726283, 1179, 249, 4.566602745850665],
                   [64.20504545453741, 1187, 353, 4.379298055599278],
                   [340.12495454530836, 1188, 369, 4.555277120157677],
                   [112.01540909091636, 424, 37, 4.477070266952174],
                   [506.63027272729323, 485, 2, 7.059618162802378]]

ode_100_profits = [[543.1144545452498, 520, 99, 10.726769903818516],
                   [509.53390909091286, 132, 32, 9.878543578136602],
                   [601.7456818181818, 227, 75, 11.171175411806098],
                   [205.3636818180821, 511, 104, 11.532180955065943],
                   [276.28095454539834, 344, 76, 11.093708356929406],
                   [160.4645454545598, 42, 38, 9.432566115199032],
                   [113.80886363637364, 32, 20, 14.961562856145186],
                   [143.14227272729477, 71, 49, 12.261880416869353]]

ode_50_profits = [[684.5629545454149, 266, 356, 6.820274216798373],
                  [163.5214090909476, 95, 320, 6.436579392113242],
                  [551.0744545454872, 139, 284, 7.014328339674776],
                  [443.7559545455042, 123, 379, 8.469203022999121],
                  [368.70413636361263, 212, 440, 7.859655563597787],
                  [427.558136363592, 202, 357, 7.413769389588828],
                  [466.8189545454379, 102, 302, 7.844850286261812],
                  [-9.061590909097049, 174, 3360, 7.377469056282828]]

ode_20_profits = [[128.8821818181841, 9, 68, 3.686316375062458],
                  [196.71577272728246, 54, 69, 5.499894756859591],
                  [307.253863636357, 150, 72, 4.2335169955998415],
                  [137.37486363637072, 162, 123, 5.120399224624745],
                  [106.19972727273762, 243, 96, 4.5974501293307615],
                  [119.62736363635031, 62, 68, 5.3018753518294295],
                  [59.95772727272333, 36, 53, 4.166107332972407],
                  [40.41522727272877, 52, 67, 4.738099442084597]]

def calc_norms(profits):
    norms = []
    for prof in profits:
        norms.append(prof[0] / (prof[1] + prof[2]))

    return norms

def calc_t_stats(profits):
    s = statistics.stdev([i[0] for i in profits])
    mu = statistics.mean([i[0] for i in profits])
    t_stats = []
    for prof in profits:
        total = prof[1] + prof[2]
        mu = prof[0] / total
        t_stats.append((mu * math.sqrt(total)) / (prof[3]))
    
    return t_stats

def box_plot(df, title, xlabel, ylabel):
    sns.set_theme(style='darkgrid')
    ax = sns.boxplot(data=df)
    ax.set_title(title)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.plot()
    plt.show()

def compare(dict_norm, dict_t, experiment, xlabel):
    df_norm = pd.DataFrame(dict_norm)
    df_t = pd.DataFrame(dict_t)
    
    box_plot(df_norm, experiment + " Normalised Profits", xlabel, "GBP per trade")
    box_plot(df_t, experiment + " t-statistics", xlabel, "t-statistic")

dict_n_lstm = {'20': calc_norms(lstm_20_profits),
               '50': calc_norms(lstm_50_profits),
               '100': calc_norms(lstm_100_profits)}
dict_t_lstm = {'20': calc_t_stats(lstm_20_profits),
               '50': calc_t_stats(lstm_50_profits),
               '100': calc_t_stats(lstm_100_profits),}
# compare(dict_n_lstm, dict_t_lstm, "LSTM", "Prediction Horizon")

dict_n_ode = {'20': calc_norms(ode_20_profits),
               '50': calc_norms(ode_50_profits),
               '100': calc_norms(ode_100_profits)}
dict_t_ode = {'20': calc_t_stats(ode_20_profits),
               '50': calc_t_stats(ode_50_profits),
               '100': calc_t_stats(ode_100_profits),}
# compare(dict_n_ode, dict_t_ode, "ODE", "Prediction Horizon")

dict_n_100 = {'ODE': calc_norms(ode_100_profits),
              'LSTM': calc_norms(lstm_100_profits)}
dict_t_100 = {'ODE': calc_t_stats(ode_100_profits),
              'LSTM': calc_t_stats(lstm_100_profits)}
# compare(dict_n_100, dict_t_100, "100 Horizon", "Model")

dict_n_50 = {'ODE': calc_norms(ode_50_profits),
             'LSTM': calc_norms(lstm_50_profits)}
dict_t_50 = {'ODE': calc_t_stats(ode_50_profits),
             'LSTM': calc_t_stats(lstm_50_profits)}
# compare(dict_n_50, dict_t_50, "50 Horizon", "Model")

dict_n_20 = {'ODE': calc_norms(ode_20_profits),
             'LSTM': calc_norms(lstm_20_profits)}
dict_t_20 = {'ODE': calc_t_stats(ode_20_profits),
             'LSTM': calc_t_stats(lstm_20_profits)}
# compare(dict_n_20, dict_t_20, "20 Horizon", "Model")

for elem in ode_100_profits:
    print(elem[0] / (elem[1] + elem[2]))
print("hello")
for elem in lstm_100_profits:
    print(elem[0] / (elem[1] + elem[2]))


