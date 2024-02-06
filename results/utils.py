import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def get_lists_to_plot(dir):
  with open(os.path.join(dir, "train/total_rewards"), 'rb') as f:
    rews = pickle.load(f)
  with open(os.path.join(dir, "train/total_wins"), 'rb') as f:
    wins = pickle.load(f)
  return rews, wins

def plot_error_lineplots(list_infos, n_average, xlabel="episode_group_av", ylabel="rewards", names=["dqn", "ddqn", "ddqn2"]):
  sizes = [len(list_info) for list_info in list_infos]
  cutting_th = min(sizes)
  combined_df = []
  for j, list_info in enumerate(list_infos):
    list_info = list_info[:cutting_th]

    x = np.asarray(list_info)
    df = pd.DataFrame(list(zip(range(0, len(x)), x)), columns=['episode', 'info'])
    new_df = pd.DataFrame()
    subdf_n_average = []#n episodes will be one item here
    for i in range(len(x)-n_average+1):
      #print(df[i: i+n_average]["reward"].values)
      data = {xlabel: np.ones(n_average)*i,
              ylabel: df[i: i+n_average]["info"].values}
      sub_df = pd.DataFrame(data)
      subdf_n_average.append(sub_df)
    new_df = pd.concat(subdf_n_average)
    new_df["Model"] = names[j]
    combined_df.append(new_df)
  combined_df = pd.concat(combined_df)
  #Plot
  sns.set(style="darkgrid")
  sns.lineplot(x=xlabel, y=ylabel, data=combined_df, ci="sd", hue="Model")

def plot_average_reward(data, title="Episodes trained vs. Average Rewards", n_average=500):
    plt.title(title)
    plt.plot([0 for _ in range(n_average)] + 
            np.convolve(data, np.ones((n_average,))/n_average, mode="valid").tolist())
    plt.show()
        
# def plot_error_lineplot(list_info, n_average, xlabel="episode_group_av", ylabel="rewards"):
#   x = np.asarray(list_info)
#   df = pd.DataFrame(list(zip(range(0, len(x)), x)), columns=['episode', 'info'])
#   new_df = pd.DataFrame()
#   subdf_n_average = []#n episodes will be one item here
#   for i in range(len(x)-n_average+1):
#     #print(df[i: i+n_average]["reward"].values)
#     data = {xlabel: np.ones(n_average)*i,
#             ylabel: df[i: i+n_average]["info"].values}
#     sub_df = pd.DataFrame(data)
#     subdf_n_average.append(sub_df)
#   new_df = pd.concat(subdf_n_average, ignore_index=True)

#   #Plot
#   sns.set(style="darkgrid")
#   sns.lineplot(x="episode_group_av", y="rewards", data=new_df, ci="sd")