from os import sep
import matplotlib.pyplot as plt
import glob
import natsort
import pandas as pd
import japanize_matplotlib

loss_log_list = glob.glob("loss_log_*.txt")
loss_log_list = natsort.natsorted(loss_log_list)

for i, loss_log in enumerate(loss_log_list):
    df = pd.read_csv(loss_log, sep="\t", header=None)
    df = df.set_axis(["step", "accuracy", "reward", "loss"], axis="columns")
    plt.plot(df["step"], df["accuracy"], label=f"{i + 1:2d}回目の試行")

plt.xlabel("Learning Step")
plt.ylabel("Accuracy")
plt.ylim((0, 1))
plt.legend(bbox_to_anchor=(1.025, 1), loc='upper left', borderaxespad=0)
plt.savefig("accuracy.png", bbox_inches="tight", pad_inches=0.05)
