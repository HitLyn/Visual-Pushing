import os
from tensorboard.backend.event_processing import event_accumulator
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython import embed


def load_tf(job):
    ea = event_accumulator.EventAccumulator(job, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    df = {}
    # m_names = ea.Tags()['scalars']
    m_names = ['eval/success_rate']
    # embed()
    for n in m_names:
        df[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "step", n])
        df[n].drop("wall_time", axis=1, inplace=True)
        df[n] = df[n].set_index("step")
        print('loading...')
    return pd.concat([v for k, v in df.items()], axis=1)


def load_tf_jobs():
    rows = []
    # embed()
    for job in job_dirs:
        results = load_tf(job)
        rows.append(results)
    # embed()
    return rows


def get_data_frame(df_list, num_list):
    list_index = []
    df_list_to_plot = []
    for num in num_list:
        for i in range(4):
            list_index.append(num * 4 + i)
    # concatnate df to plot
    for index in list_index:
        df = df_list[index]
        # embed()
        dict_data = {'tag': [job_dirs[index].split('/')[7]] * 950, 'step': np.linspace(2450,2409950,num = 950)/1e6,
                     'success': df['eval/success_rate'][:950]}
        df_new = pd.DataFrame(data=dict_data)
        # embed()
        df_new['average_success_rate'] = df_new.success.rolling(50, min_periods=1).mean()
        # df_new['average_success_rate'] = df_new.success
        # embed()
        df_list_to_plot.append(df_new)
        print('tag: ', job_dirs[index].split('/')[7])

    df_to_plot = pd.concat(df_list_to_plot)
    return df_to_plot


if __name__ == "__main__":
    push_home = os.environ["VISUAL_PUSHING_HOME"]
    job_dirs = glob.glob(f"{push_home}/log_files/final_data/*/*/event*")
    # embed()
    df_list = load_tf_jobs()
    line_name_list = [job.split('/')[7] for job in job_dirs[::4]]
    print(line_name_list)
    embed()
    # df = get_data_frame(df_list, [1,2,3,5,6,8])
    # plot
    df1 = get_data_frame(df_list, [0,7,4])
    df2 = get_data_frame(df_list, [1,2,8])
    df3 = get_data_frame(df_list, [3,5,6])

    fig = plt.figure(figsize=(15, 30))
    # 1
    ax1 = plt.subplot2grid((18, 8), (0, 0), colspan=8, rowspan=5)
    # ax1 = plt.subplot(1,3,1)
    sns.lineplot(data=df1, x="step", y="average_success_rate", hue='tag')
    # ax1.set(xlabel = None, ylabel = 'Success rate')
    # ax1.xaxis.set_visible(False)
    ax1.set_xlabel('Steps (M)', fontsize=28)
    ax1.set_ylabel('Success rate', fontsize=28)
    ax1.set_title('Latent Space Dimension', fontdict={'fontsize': 30, 'fontweight': "medium"}, pad=5.0)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 1)
    # ax1.set_xticks([])
    ax1.set_facecolor('gainsboro')
    ax1.grid(color='white')
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.legend(frameon=False, loc='upper left', ncol=1, prop={'size': 28})

    # # 2
    ax2 = plt.subplot2grid((18, 8), (6, 0), colspan=8, rowspan=5)
    # ax2 = plt.subplot(1, 3, 2)
    sns.lineplot(data=df2, x="step", y="average_success_rate", hue='tag')
    ax2.set_title('Reward Type', fontdict={'fontsize': 30, 'fontweight': "medium"}, pad=1.0)
    # ax2.set(xlabel = None, ylabel = 'Success rate')
    ax2.set_ylabel('Success rate', fontsize=28)
    ax2.set_xlabel('Steps (M)', fontsize=28)
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 1)
    ax2.grid(color='white')
    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.set_facecolor('gainsboro')
    # ax2.set_xticks([])
    ax2.legend(frameon=False, loc='upper left', ncol=1, prop={'size': 28})
    #
    # # 3
    ax3 = plt.subplot2grid((18, 8), (12, 0), colspan=8, rowspan=5)
    # ax3 = plt.subplot(1, 3, 3)
    sns.lineplot(data=df3, x="step", y="average_success_rate", hue='tag')
    ax3.set_title('Input Modality', fontdict={'fontsize': 30, 'fontweight': "medium"}, pad=1.0)
    # ax3.set(xlabel = 'Eipsodes', ylabel = 'Success rate', fontsize = 12)
    ax3.set_xlabel('Steps (M)', fontsize=28)
    ax3.set_ylabel('Success rate', fontsize=28)
    ax3.set_xlim(0, 2.5)
    ax3.set_ylim(0, 1)
    ax3.grid(color='white')
    ax3.tick_params(axis='both', which='major', labelsize=28)
    ax3.legend(frameon=False, loc='upper left', ncol=1, prop={'size': 28})
    ax3.set_facecolor('gainsboro')
    # #

    plt.tight_layout(pad=1, h_pad=5.0)
    plt.savefig('curve_vertival.pdf')
    plt.savefig('curve_vertival.png')
    # plt.show()