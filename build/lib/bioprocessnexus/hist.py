import tkinter as tk
import threading
import pickle
from datetime import datetime
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from distfit import distfit
from .helpers import *


def init_hist(parent):
    # List of distribution names for distfit
    parent.distfit_distributions = ["alpha", "beta", "gamma", "loggamma", "laplace", "chi2",
                                    "norm", "lognorm", "exponnorm", "gennorm", "powernorm", "t",
                                    "uniform", "weibull", "triang", "levy", "dweibull",
                                    "expon", "bradford", "arcsine", "burr"]
    parent.n_plots = 0
    # Checking if a model has been loaded and responses and features have been selected
    if hasattr(parent, "y_names") is False:
        tk.messagebox.showerror("Error message", "No model has been loaded or " +
                                "responses and features have not been selected.")
    else:
        check_hist_queue(parent)
        threading.Thread(target=plot_hist(parent)).start()


def check_hist_queue(parent):
    while not parent.queue.empty():
        # Creating a new window for the histogram
        parent.histogram_window = tk.Toplevel(parent)
        parent.histogram_window.grid_rowconfigure(0, weight=1)
        parent.histogram_window.grid_columnconfigure(0, weight=1)
        parent.histogram_window.title("Histogram")
        # Getting the figure from the queue
        fig = parent.queue.get_nowait()
        canvas = FigureCanvasTkAgg(fig, master=parent.histogram_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)
        parent.n_plots += 1
    # Checking if all histograms have been plotted
    if parent.n_plots != len(parent.y_names):
        parent.after(50, check_hist_queue, parent)


def plot_hist(parent):
    # Checking if a model has been loaded
    if parent.model_loaded == 0:
        parent.model_dir = str.rsplit(parent.filename, "/", 1)[0]+"/data/no_model_" + \
            datetime.now().strftime("%m_%d_%Y_%H_%M")

    if parent.gr_plots_switch_var.get() == "1":
        alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                    "T", "U", "V", "W", "X", "Y", "Z"]
        n_subplots = len(parent.y_names)
        subplots_per_row = 3
        n_rows = (n_subplots + subplots_per_row - 1) // subplots_per_row
        if n_subplots > 3:
            fig, axs = plt.subplots(n_rows, subplots_per_row, figsize=(15, 10))
        else:
            fig, axs = plt.subplots(n_rows, subplots_per_row, figsize=(15, 5))
        axs = axs.flatten()
        for i in range(n_subplots, len(axs)):
            fig.delaxes(axs[i])

    subplot_counter = 0
    for y_dir in parent.y_names:
        if parent.model_loaded == 1:
            # Load y_train and y_test
            with open(f"{parent.model_dir}/{y_dir}/y_test", "rb") as f:
                y_test = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/y_train", "rb") as f:
                y_train = pickle.load(f)
            # Merge them
            y_hist = np.hstack((y_train, y_test))
            # Load normalization parameters
            with open(f"{parent.model_dir}/{y_dir}/y_mus", "rb") as f:
                y_mus = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/y_stds", "rb") as f:
                y_stds = pickle.load(f)
            # Denormalize
            y_hist = denormalize(y_hist, y_mus, y_stds)
            x_dist = np.linspace(y_hist.min(), y_hist.max(), 1000)
        else:
            try:
                y_hist = np.array(parent.data.loc[:, y_dir.replace("_", " ")])
            except:
                y_hist = np.array(parent.data.loc[:, y_dir])
            x_dist = np.linspace(y_hist.min(), y_hist.max(), 1000)

        # Fitting the distribution
        dfit = distfit(distr=parent.distfit_distributions)
        dfit.fit_transform(y_hist)
        param = dfit.model["params"]
        dist = getattr(st, dfit.model["name"])

        # Calculating the probability density function
        if len(param) == 2:
            y_dist = dist.pdf(x_dist,
                              param[0],
                              param[1])
        if len(param) == 3:
            y_dist = dist.pdf(x_dist,
                              param[0],
                              param[1],
                              param[2])
        if len(param) == 4:
            y_dist = dist.pdf(x_dist,
                              param[0],
                              param[1],
                              param[2],
                              param[3])

        # Saving histogram parameters to a file
        save_dir = check_dir(parent, y_dir, "logs")
        with open(save_dir+"/histogram_parameters.txt", 'w', encoding="utf-8") as f:
            for i in dfit.model:
                f.write(f"{i}: {dfit.model[i]}\n")

        # Annotating the maximum point on the plot
        if x_dist[y_dist.argmax()] > 1:
            max_annotation = "%.2f" % round(x_dist[y_dist.argmax()], 2)
        else:
            max_annotation = "%.5f" % round(x_dist[y_dist.argmax()], 2)
        model_name = dfit.model["name"]

        if parent.gr_plots_switch_var.get() != "1":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_title(
                f"Distribution of {pretty_y}\nDistribution={model_name}")
        else:
            ax = axs[subplot_counter]
            ax.text(-0.1, 1.1,
                    alphabet[subplot_counter],
                    fontsize=20,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
            subplot_counter += 1
        pretty_y = y_dir.replace("_", " ")
        ax.hist(y_hist, bins=100)
        ax.set_xlabel(f"{pretty_y}", fontsize=15)
        ax.set_ylabel("Number of samples", fontsize=15)
        ax2 = ax.twinx()
        ax2.plot(x_dist, y_dist, color="red")
        ax2.vlines(x_dist[y_dist.argmax()], 0, y_dist.max(),
                   color="red", linestyle="--")
        ax2.set_ylabel("Probability density function", fontsize=15)
        ax2.set_ylim(0)
        ax2.annotate(f"{max_annotation}",
                     (x_dist[y_dist.argmax()]+x_dist[y_dist.argmax()]*0.05,
                      y_dist.max()*1.005),
                     color="red",
                     fontsize=13)
        plt.tight_layout()
        # Saving the plot as an image
        img_dir = check_dir(parent, y_dir, "images")
        if parent.gr_plots_switch_var.get() != "1":
            fig.savefig(f"{img_dir}/histogram.png")
            parent.queue.put(fig)
    if parent.gr_plots_switch_var.get() == "1":
        fig.tight_layout()
        img_dir = img_dir.rsplit("/", 1)[0]
        fig.savefig(f"{img_dir}/histogram.png")
        parent.queue.put(fig)
    else:
        img_dir = img_dir.rsplit("/", 1)[0]
    save_dir = save_dir.rsplit("/", 1)[0]
    tk.messagebox.showinfo(
        "Information", f"Images saved at {img_dir}\nLogs saved at {save_dir}")

