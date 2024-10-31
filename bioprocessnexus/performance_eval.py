import tkinter as tk
import threading
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from joblib import load
from helpers import *


def init_plot_predictions(parent):
    if parent.model_loaded == 0:
        tk.messagebox.showerror("Error message", "No model has been loaded.")
    else:
        parent.n_plots = 0
        check_plot_predictions_queue(parent)
        threading.Thread(target=plot_predictions(parent)).start()


def check_plot_predictions_queue(parent):
    while not parent.queue.empty():
        parent.evaluation_window = tk.Toplevel(parent)
        parent.evaluation_window.grid_rowconfigure(0, weight=1)
        parent.evaluation_window.grid_columnconfigure(0, weight=1)
        parent.evaluation_window.title("Evaluate model predidctions")
        fig = parent.queue.get_nowait()
        canvas = FigureCanvasTkAgg(fig, master=parent.evaluation_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)
        parent.n_plots += 1
    if parent.n_plots != len(parent.y_names):
        parent.after(50, check_plot_predictions_queue, parent)


def plot_predictions(parent):
    if parent.gr_plots_switch_var.get() == "1":
        alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                    "T", "U", "V", "W", "X", "Y", "Z"]
        subplot_counter = 0
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

    for y_dir in parent.y_names:
        # Load model
        model_type_dir = f"{parent.model_dir}/{y_dir}"
        model_type = [filename for filename in os.listdir(
            model_type_dir) if filename.endswith(".joblib")][0]
        model = load(f"{parent.model_dir}/{y_dir}/{model_type}")
        # Load X_test and y_test
        with open(f"{parent.model_dir}/{y_dir}/X_test", "rb") as f:
            X_test = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_test", "rb") as f:
            y_test = pickle.load(f)
        # Load normalization parameters
        with open(f"{parent.model_dir}/{y_dir}/y_mus", "rb") as f:
            y_mus = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_stds", "rb") as f:
            y_stds = pickle.load(f)
        # Make model predictions
        preds = model.predict(X_test)
        # Reshape in case the model output is not a 1D array
        if len(preds.shape) != 1:
            preds = preds[:, 0]
        # Denormalize
        preds = denormalize(preds, y_mus, y_stds)
        y_test = denormalize(y_test, y_mus, y_stds)
        RMSE = (np.mean((y_test-preds)**2))**0.5
        NRMSE = RMSE*100/(np.max(y_test)-np.min(y_test))
        plot_min = np.hstack((preds, y_test)).min()
        plot_max = np.hstack((preds, y_test)).max()

        pretty_y = y_dir.replace("_", " ")
        if parent.gr_plots_switch_var.get() != "1":
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            ax = axs[subplot_counter]
            ax.text(-0.1, 1.1,
                    alphabet[subplot_counter],
                    fontsize=20,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes)
            subplot_counter += 1

        ax.scatter(preds, y_test)
        ax.set_xlim(plot_min*0.95, plot_max*1.05)
        ax.set_ylim(plot_min*0.95, plot_max*1.05)
        ax.plot((plot_min*0.95, plot_max*1.05), (plot_min*0.95, plot_max*1.05),
                linestyle="-",
                color="red")
        ax.set_xlabel(f"Predicted {pretty_y}", fontsize=15)
        ax.set_ylabel(f"{pretty_y}", fontsize=15)
        img_dir = check_dir(parent, y_dir, "images")
        if parent.gr_plots_switch_var.get() != "1":
            fig.savefig(f"{img_dir}/performance.png")
            parent.queue.put(fig)

    dir = parent.model_dir.replace("data", "images")
    if parent.gr_plots_switch_var.get() == "1":
        fig.tight_layout()
        fig.savefig(f"{dir}/performance.png", dpi=600)
        parent.queue.put(fig)

    tk.messagebox.showinfo("Information", f"Images saved at {dir}")

