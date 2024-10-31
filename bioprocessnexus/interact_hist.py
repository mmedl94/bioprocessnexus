import tkinter as tk
from datetime import datetime
import pickle
from functools import partial
import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RangeSlider
import matplotlib.gridspec as gridspec
import customtkinter as ctk
from distfit import distfit
import scipy.stats as st
from helpers import *


def interactive_hist_welcome(parent):
    parent.distfit_distributions = ["alpha", "beta", "gamma", "loggamma", "laplace", "chi2",
                                    "norm", "lognorm", "exponnorm", "gennorm", "powernorm", "t",
                                    "uniform", "weibull", "triang", "levy", "dweibull",
                                    "expon", "bradford", "arcsine", "burr"]

    if hasattr(parent, "y_names") is False:
        tk.messagebox.showerror("Error message", "No model has been loaded or " +
                                "responses and features have not been selected.")
    else:
        parent.interactive_hist_welc_window = tk.Toplevel(parent)
        parent.interactive_hist_welc_window.title("Interactive histogram")
        parent.inter_hist_welc_label = ctk.CTkLabel(parent.interactive_hist_welc_window,
                                                    text="Choose a response",
                                                    font=ctk.CTkFont(size=17, weight="bold"))
        parent.inter_hist_welc_label.grid(row=0, column=0, padx=20, pady=15)
        row_counter = 1
        for y_dir in parent.y_names:
            parent.welc_button = ctk.CTkButton(parent.interactive_hist_welc_window,
                                               width=parent.button_width,
                                               text=f"{y_dir}",
                                               command=partial(interactive_hist,
                                                               parent=parent,
                                                               y_dir=y_dir))
            parent.welc_button.grid(row=row_counter, column=0, padx=20, pady=5)
            row_counter += 1


def interactive_hist(parent, y_dir):
    try:
        parent.interactive_hist_window.destroy()
    except:
        pass
    try:
        del parent.ax4
    except:
        pass
    parent.interactive_hist_window = tk.Toplevel(parent)
    parent.interactive_hist_window.title("Interactive histogram")
    parent.image_frame = ctk.CTkScrollableFrame(parent.interactive_hist_window)
    parent.image_frame.grid(row=0, column=0, columnspan=5)
    parent.warning_issued = 0
    parent.warning_trigger = 0
    if parent.model_loaded == 1:
        # Load y_train and y_test
        with open(f"{parent.model_dir}/{y_dir}/y_test", "rb") as f:
            y_test = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_train", "rb") as f:
            y_train = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_test", "rb") as f:
            X_test = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_train", "rb") as f:
            X_train = pickle.load(f)
        # Megre them
        y_hist = np.hstack((y_train, y_test))
        X_hist = np.vstack((X_train, X_test))
        # Load normalization parameters
        with open(f"{parent.model_dir}/{y_dir}/y_mus", "rb") as f:
            y_mus = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_stds", "rb") as f:
            y_stds = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_mus", "rb") as f:
            X_mus = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_stds", "rb") as f:
            X_stds = pickle.load(f)
        with open(f"{parent.model_dir}/feature_selection.pkl", "rb") as f:
            parent.X_names = pickle.load(f)
        parent.X_hist = X_hist*X_stds+X_mus
        parent.y_hist = y_hist*y_stds+y_mus
    else:
        parent.model_dir = str.rsplit(parent.filename, "/", 1)[0]+"/data/no_model_" + \
            datetime.now().strftime("%m_%d_%Y_%H_%M")
        parent.X_names = list(np.array(list(parent.response_selection))[
                              parent.feature_bool_vars])
        try:
            parent.y_hist = np.array(
                parent.data.loc[:, y_dir.replace("_", " ")])
        except:
            parent.y_hist = np.array(parent.data.loc[:, y_dir])
        parent.X_hist = np.array(parent.data.iloc[:, parent.feature_bool_vars])
    parent.y_name = y_dir
    # Initialize upd_y_hist
    parent.upd_y_hist = parent.y_hist
    parent.max_pdf = 0
    # Get the vertical ratio of ax1. It is set so that the histogram height is consistant
    vertical_ratio_ax1 = parent.X_hist.shape[1]//15
    if vertical_ratio_ax1 == 0:
        vertical_ratio_ax1 = 1
    if parent.X_hist.shape[1]*0.3 < 7:
        parent.fig = plt.figure(figsize=(13, 6))
    else:
        parent.fig = plt.figure(figsize=(13, parent.X_hist.shape[1]*0.3))

    gs = gridspec.GridSpec(vertical_ratio_ax1, 5, figure=parent.fig)
    parent.ax1 = parent.fig.add_subplot(gs[0, :2])
    ax2 = parent.fig.add_subplot(gs[:, 2:])
    parent.ax1.hist(parent.y_hist, bins=100)
    pretty_y = y_dir.replace("_", " ")
    parent.ax1.set_xlabel(f"{pretty_y}", fontsize=15)
    parent.ax1.set_ylabel("Number of samples", fontsize=15)
    # Make twin axis for dist fit and clear it
    parent.ax3 = parent.ax1.twinx()
    parent.ax3.tick_params(axis="y",
                           which="both",
                           bottom=False,
                           top=False,
                           right=False,
                           labelright=False)
    ax2.axis("off")

    parent.sliders = []
    intervalls = np.linspace(0.95, 0.05, parent.X_hist.shape[1])
    for i in range(parent.X_hist.shape[1]):
        slider_ax = parent.fig.add_axes([0.75, intervalls[i], 0.10, 0.025])
        slider = RangeSlider(slider_ax, f"{parent.X_names[i]}  ",
                             valmin=parent.X_hist.min(axis=0)[i]*0.999,
                             valmax=parent.X_hist.max(axis=0)[i]*1.001,
                             valinit=(parent.X_hist.min(axis=0)[i]*0.999, parent.X_hist.max(axis=0)[i]*1.001))
        slider.on_changed(partial(update_hist, parent=parent))
        parent.sliders.append(slider)
    parent.fig.canvas.draw_idle()
    plt.tight_layout()
    parent.canvas = FigureCanvasTkAgg(parent.fig, parent.image_frame)
    parent.canvas.draw()
    parent.canvas.get_tk_widget().grid(row=0, column=0)

    parent.image_frame.configure(width=1100*parent.scaling_factor,
                                 height=500*parent.scaling_factor)

    parent.fit_dist_button = ctk.CTkButton(parent.interactive_hist_window,
                                           width=parent.button_width,
                                           text="Fit distribution",
                                           command=lambda: fit_dist(parent))
    parent.fit_dist_button.grid(row=1, column=0, padx=20, pady=5)
    parent.clr_dist_button = ctk.CTkButton(parent.interactive_hist_window,
                                           width=parent.button_width,
                                           text="Clear distribution",
                                           command=lambda: clear_dist(parent))
    parent.clr_dist_button.grid(row=2, column=0, padx=20, pady=5)
    parent.save_button = ctk.CTkButton(parent.interactive_hist_window,
                                       width=parent.button_width,
                                       text="Save image",
                                       command=lambda: save_inter_hist(parent))
    parent.save_button.grid(row=3, column=0, padx=20, pady=5)

    parent.lower_b_label = ctk.CTkLabel(parent.interactive_hist_window,
                                        text="Lower bound: ")
    parent.lower_b_label.grid(row=2, column=1, padx=20, pady=5)

    parent.upper_b_label = ctk.CTkLabel(parent.interactive_hist_window,
                                        text="Upper bound: ")
    parent.upper_b_label.grid(row=3, column=1, padx=20, pady=5)

    parent.probability_label = ctk.CTkLabel(parent.interactive_hist_window,
                                            text="Probability: ")
    parent.probability_label.grid(row=4, column=1, padx=20, pady=5)

    parent.get_prob_button = ctk.CTkButton(parent.interactive_hist_window,
                                           width=parent.button_width,
                                           text="Calculate probability",
                                           command=lambda: get_probability(parent))
    parent.get_prob_button.grid(row=1, column=1, padx=20, pady=5, columnspan=2)

    parent.lower_b_var = tk.StringVar()
    parent.upper_b_var = tk.StringVar()
    parent.probability_var = tk.StringVar()

    parent.lower_b_entry = tk.Entry(parent.interactive_hist_window,
                                    textvariable=parent.lower_b_var)
    parent.lower_b_entry.grid(row=2, column=2, padx=20, pady=5)

    parent.upper_b_entry = tk.Entry(parent.interactive_hist_window,
                                    textvariable=parent.upper_b_var)
    parent.upper_b_entry.grid(row=3, column=2, padx=20, pady=5)

    parent.probability_entry = tk.Entry(parent.interactive_hist_window,
                                        bg="grey",
                                        state="disabled",
                                        textvariable=parent.probability_var)
    parent.probability_entry.grid(row=4, column=2, padx=20, pady=5)


def save_inter_hist(parent):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    fig_for_saving = copy.copy(parent.fig)
    new_manager.canvas.figure = fig_for_saving
    fig_for_saving.set_canvas(new_manager.canvas)
    fig_for_saving.show()
    img_dir = check_dir(parent, parent.y_name, "images")
    fig_for_saving.savefig(f"{img_dir}/interactive_hist_snapshot.png", dpi=300)
    save_dir = check_dir(parent, parent.y_name, "logs")
    tk.messagebox.showinfo(
        "Information", f"Images saved at {img_dir}\nLog saved at {save_dir}")
    save_dir = f"{save_dir}/interactive_hist_log.txt"
    try:
        with open(save_dir, 'w', encoding="utf-8") as f:
            for i in parent.dfit.model:
                f.write(f"{i}: {parent.dfit.model[i]}\n")
    except:
        pass


def clear_dist(parent):
    parent.ax3.set_title("")
    parent.max_pdf = 0
    parent.ax3.tick_params(axis='y',
                           which='both',
                           bottom=False,
                           top=False,
                           right=False,
                           labelright=False)
    try:
        del parent.dist_func
    except:
        pass
    parent.ax3.clear()
    try:
        parent.ax4.clear()
    except:
        pass
    parent.fig.canvas.draw()


def get_probability(parent):
    if hasattr(parent, "dist_func") is False:
        tk.messagebox.showerror("Error message", "Please fit probability distribution first.",
                                parent=parent.interactive_hist_window)
    else:
        try:
            lower_cdf = parent.dist_func.cdf(float(parent.lower_b_var.get()))
            upper_cdf = parent.dist_func.cdf(float(parent.upper_b_var.get()))
        except:
            tk.messagebox.showerror("Error message", "Please enter valid values for the lower and upper bound.",
                                    parent=parent.interactive_hist_window)

    probability = upper_cdf-lower_cdf
    parent.probability_var.set(probability)
    x_dist = np.linspace(float(parent.lower_b_var.get()),
                         float(parent.upper_b_var.get()))
    try:
        parent.ax4.clear()
    except:
        parent.ax4 = parent.ax1.twinx()
    parent.ax4.tick_params(axis='y',
                           which='both',
                           bottom=False,
                           top=False,
                           right=False,
                           left=False,
                           labelright=False,
                           labelleft=False)
    parent.ax4.set_ylim(0, parent.y_lim_max)
    parent.ax4.fill_between(x_dist,
                            parent.dist_func.pdf(x_dist),
                            color='r',
                            alpha=0.5)
    parent.interactive_hist_window.update()
    parent.fig.canvas.draw()


def fit_dist(parent):
    dfit = distfit(distr=parent.distfit_distributions)
    dfit.fit_transform(parent.upd_y_hist)
    param = dfit.model["params"]
    dist = getattr(st, dfit.model["name"])
    parent.dist_param = param
    x_dist = np.linspace(parent.y_hist.min(), parent.y_hist.max(), 1000)

    if len(param) == 2:
        y_dist = dist.pdf(x_dist,
                          param[0],
                          param[1])
        parent.dist_func = dist(param[0],
                                param[1])
    if len(param) == 3:
        y_dist = dist.pdf(x_dist,
                          param[0],
                          param[1],
                          param[2])
        parent.dist_func = dist(param[0],
                                param[1],
                                param[2])
    if len(param) == 4:
        y_dist = dist.pdf(x_dist,
                          param[0],
                          param[1],
                          param[2],
                          param[3])
        parent.dist_func = dist(param[0],
                                param[1],
                                param[2],
                                param[3])

    if x_dist[y_dist.argmax()] > 1:
        max_annotation = "%.2f" % round(x_dist[y_dist.argmax()], 2)
    else:
        max_annotation = "%.5f" % round(x_dist[y_dist.argmax()], 2)
    parent.ax3.plot(x_dist, y_dist, color="red")
    parent.ax3.tick_params(axis='y',
                           which='both',
                           bottom=True,
                           top=True,
                           right=True,
                           labelright=True)
    parent.ax3.vlines(x_dist[y_dist.argmax()], 0,
                      y_dist.max(), color="red", linestyle="--")
    parent.ax3.yaxis.set_label_position("right")
    parent.ax3.yaxis.tick_right()
    parent.ax3.set_ylabel("Probability density function", fontsize=15)
    if parent.max_pdf < y_dist.max():
        parent.max_pdf = y_dist.max()
    parent.ax3.set_ylim(0, parent.max_pdf.max()*1.1)
    parent.y_lim_max = parent.max_pdf.max()*1.1
    parent.ax3.annotate(f"{max_annotation}",
                        (x_dist[y_dist.argmax()], y_dist.max()*1.02),
                        color="red",
                        fontsize=13)
    model_name = dfit.model["name"]
    parent.ax3.set_title(f"Distribution: {model_name}")
    plt.tight_layout()
    parent.fig.canvas.draw()
    parent.dfit = dfit


def update_hist(val, parent):
    upd_y_hist = parent.y_hist.copy()
    upd_X_hist = parent.X_hist.copy()
    for i in range(len(parent.sliders)):
        indices_to_keep = np.where((upd_X_hist[:, i] > [parent.sliders[i].val[0]]) &
                                   (upd_X_hist[:, i] < [parent.sliders[i].val[1]]))
        upd_y_hist = upd_y_hist[indices_to_keep[0]]
        upd_X_hist = upd_X_hist[indices_to_keep[0]]

    if len(indices_to_keep[0]) < 500:
        parent.warning_trigger = 1
    else:
        parent.warning_trigger = 0

    parent.upd_y_hist = upd_y_hist
    parent.ax1.cla()
    parent.ax1.hist(upd_y_hist, bins=100)
    pretty_y = parent.y_name.replace("_", " ")
    parent.ax1.set_xlabel(f"{pretty_y}", fontsize=15)
    parent.ax1.set_ylabel("Number of samples", fontsize=15)
    parent.ax1.set_xlim(parent.y_hist.min(), parent.y_hist.max())

    if parent.warning_trigger == 1 and parent.warning_issued == 0:
        tk.messagebox.showwarning("Warning message", "Number of samples is low.",
                                  parent=parent.interactive_hist_window)
        parent.warning_issued = 1
    if parent.warning_trigger == 0 and parent.warning_issued == 1:
        parent.warning_issued = 0
    parent.fig.canvas.draw_idle()

