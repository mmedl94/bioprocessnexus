import tkinter as tk
import os
import pickle
import threading
import numpy as np
import customtkinter as ctk
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .model_training import PLS_optimize_train
from .helpers import check_dir, denormalize


def init_data_scaling(parent):
    if parent.model_loaded == 0:
        tk.messagebox.showerror("Error message", "No model has been loaded.")
    else:
        parent.n_plots = 0
        check_data_scaling_queue(parent)
        threading.Thread(target=data_scaling(parent)).start()


def check_data_scaling_queue(parent):
    while not parent.queue.empty():
        parent.evaluation_window = tk.Toplevel(parent)
        parent.evaluation_window.grid_rowconfigure(0, weight=1)
        parent.evaluation_window.grid_columnconfigure(0, weight=1)
        parent.evaluation_window.title("Evaluate data scaling performance")
        fig = parent.queue.get_nowait()
        canvas = FigureCanvasTkAgg(fig, master=parent.evaluation_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)
        parent.n_plots += 1
    if parent.n_plots != len(parent.y_names):
        parent.after(50, check_data_scaling_queue, parent)


def data_scaling(parent):
    parent.scaling_window = tk.Toplevel(parent)
    parent.scaling_window.grid_rowconfigure(0, weight=1)
    parent.scaling_window.grid_columnconfigure(0, weight=1)
    parent.scaling_window.title("Evaluate data scaling")
    parent.figs = []

    parent.scaling_label_text = tk.StringVar()
    parent.scaling_label = ctk.CTkLabel(parent.scaling_window,
                                        textvariable=parent.scaling_label_text,
                                        font=ctk.CTkFont(
                                            size=15, weight="bold"),
                                        wraplength=1000)
    parent.scaling_label_text.set("Please enter number of evaluations")
    parent.scaling_label.grid(
        row=0, column=0, columnspan=2, padx=20, pady=(20, 10))

    parent.fraction = tk.StringVar()
    parent.fraction.set("10")
    textbox = tk.Entry(parent.scaling_window, textvariable=parent.fraction)
    textbox.grid(row=1, column=0, padx=5, pady=5)
    wait_var = tk.IntVar()
    parent.scaling_button = ctk.CTkButton(parent.scaling_window,
                                          width=parent.button_width,
                                          text="Plot scaling performance",
                                          command=lambda: wait_var.set(1))
    parent.scaling_button.grid(row=2, column=0, padx=20, pady=(20, 10))
    parent.scaling_button.wait_variable(wait_var)

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

    check_frequency = 1/int(parent.fraction.get())
    for y_dir in parent.y_names:
        # check model type
        model_type_dir = f"{parent.model_dir}/{y_dir}"
        model_type = [filename for filename in os.listdir(
            model_type_dir) if filename.endswith(".joblib")][0]

        # Load data
        with open(f"{parent.model_dir}/{y_dir}/X_test", "rb") as f:
            X_test = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_test", "rb") as f:
            y_test = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_train", "rb") as f:
            X_train = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_train", "rb") as f:
            y_train = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_mus", "rb") as f:
            y_mus = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_stds", "rb") as f:
            y_stds = pickle.load(f)

        # Denormalize data
        y_test = denormalize(y_test, y_mus, y_stds)
        y_max = np.max(y_test)
        y_min = np.min(y_test)

        RMSEs = []
        NRMSEs = []
        n_subsamples = []
        # Reduce sample size of X_train and y_train
        n_total_samples = X_train.shape[0]
        for i in range(int(1/check_frequency)):
            scaling_factor = (i+1)*check_frequency
            rand_subsample = (np.random.choice(np.arange(n_total_samples),
                                               size=int(
                                                   n_total_samples*scaling_factor),
                                               replace=False))
            X_train_sub = X_train[rand_subsample].copy()
            y_train_sub = y_train[rand_subsample].copy()

            # Train model
            if model_type == "random_forest.joblib":
                model = RandomForestRegressor()
                model.fit(X_train_sub, y_train_sub)
                preds = model.predict(X_test)

            if model_type == "gaussian_process.joblib":
                kernel = WhiteKernel(noise_level_bounds=(
                    1e-10, 1e5)) + ConstantKernel()*RBF()
                model = GaussianProcessRegressor(kernel=kernel,
                                                 random_state=0,
                                                 n_restarts_optimizer=3,
                                                 normalize_y=True).fit(X_train_sub, y_train_sub)
                preds = model.predict(X_test)

            if model_type == "partial_least_squares.joblib":
                model = PLS_optimize_train(X_train_sub, y_train_sub)
                preds = model.predict(X_test)[:, 0]
            # Denormalize data
            preds = denormalize(preds, y_mus, y_stds)

            # Evaluate model predictions
            RMSE = (np.mean((y_test-preds)**2))**0.5
            RMSEs.append(RMSE)
            NRMSEs.append(RMSE*100/(y_max-y_min))
            n_subsamples.append(int(n_total_samples*scaling_factor))

        log_data = pd.DataFrame(data={"n_subsamples": n_subsamples,
                                      "RMSE": RMSEs})
        log_dir = check_dir(parent, y_dir, "logs")
        log_data.to_csv(log_dir+"/scaling_performance.csv")

        pretty_y = y_dir.replace("_", " ")
        if parent.gr_plots_switch_var.get() != "1":
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            ax = axs[subplot_counter]
            ax.text(-0.1, 1.1,
                    alphabet[subplot_counter],
                    fontsize=20,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
            subplot_counter += 1

        ax.plot(n_subsamples, RMSEs)
        pretty_y = y_dir.replace("_", " ")
        ax.set_title(f"{pretty_y}", fontsize=15)
        ax.set_xlabel("Number of observations", fontsize=15)
        ax.set_ylabel("Root mean squared error", fontsize=15)
        ax.set_ylim(0, max(RMSEs)*1.02)

        ax2 = ax.twinx()
        ax2.set_ylabel("Normalized root mean squared error [%]", fontsize=15)
        ax2.plot(n_subsamples, NRMSEs)
        ax2.set_ylim(0, max(NRMSEs)*1.02)
        img_dir = check_dir(parent, y_dir, "images")
        if parent.gr_plots_switch_var.get() != "1":
            fig.savefig(f"{img_dir}/scaling_performance.png",
                        bbox_inches="tight")
            parent.queue.put(fig)

    log_dir = parent.model_dir.replace("data", "logs")
    img_dir = parent.model_dir.replace("data", "images")
    if parent.gr_plots_switch_var.get() == "1":
        fig.tight_layout()
        fig.savefig(f"{img_dir}/scaling_performance.png", dpi=600)
        parent.queue.put(fig)
    tk.messagebox.showinfo(
        "Information", f"Images saved at {img_dir}\n Logs saved at {log_dir}")
    parent.scaling_window.destroy()

