import os
from datetime import datetime
from functools import partial
import tkinter as tk
import pickle
import numpy as np
import scipy.stats as st
import pandas as pd
from joblib import load
import customtkinter as ctk
from helpers import *


def generate_data_interface(parent):
    if parent.model_loaded == 0:
        tk.messagebox.showerror("Error message", "No model has been loaded")
    else:
        # Load variable names
        with open(f"{parent.model_dir}/feature_selection.pkl", "rb") as f:
            parent.X_names = pickle.load(f)
        X_dim = len(parent.X_names)

        # Set boundaries for interface
        parent.upper_boundaries_display = np.full(X_dim, np.inf)
        parent.lower_boundaries_display = np.full(X_dim, -np.inf)
        parent.upper_boundaries = np.full(X_dim, np.inf)
        parent.lower_boundaries = np.full(X_dim, -np.inf)
        for y_dir in parent.y_names:
            with open(f"{parent.model_dir}/{y_dir}/min", "rb") as f:
                mins = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/max", "rb") as f:
                maxs = pickle.load(f)
            # Load normalization parameters
            with open(f"{parent.model_dir}/{y_dir}/X_mus", "rb") as f:
                X_mus = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/X_stds", "rb") as f:
                X_stds = pickle.load(f)
            # Find the minimal upper boundary
            parent.upper_boundaries_display = np.minimum(parent.upper_boundaries_display,
                                                         denormalize(maxs, X_mus, X_stds))
            # Find the maximal lower boundary
            parent.lower_boundaries_display = np.maximum(parent.lower_boundaries_display,
                                                         denormalize(mins, X_mus, X_stds))
            parent.upper_boundaries = np.minimum(parent.upper_boundaries, maxs)
            parent.lower_boundaries = np.maximum(parent.lower_boundaries, mins)

        parent.data_generation_window = tk.Toplevel(parent)
        parent.data_generation_window.title("Generate new dataset")

        height1 = X_dim*40-30
        height2 = parent.data_generation_window.winfo_screenheight()*0.7

        frame_height = np.min((height1*parent.scaling_factor,
                               height2*parent.scaling_factor))

        X_frame = ctk.CTkScrollableFrame(
            parent.data_generation_window, height=frame_height, corner_radius=0)
        X_frame.grid(row=0, column=0, sticky="ns", padx=20)

        text_for_scaling_X = []
        box_for_scaling_X = []
        parent.dist_var_dict = {}
        parent.dist_box_dict = {}
        for i in range(X_dim):
            if parent.lower_boundaries_display[i] > 1:
                lower_b = "%.2f" % parent.lower_boundaries_display[i]
            else:
                lower_b = "%.6f" % parent.lower_boundaries_display[i]

            if parent.upper_boundaries_display[i] > 1:
                upper_b = "%.2f" % parent.upper_boundaries_display[i]
            else:
                upper_b = "%.6f" % parent.upper_boundaries_display[i]
            label = ctk.CTkLabel(X_frame,
                                 text=f"{parent.X_names[i]} \nBoundaries: [{lower_b}-{upper_b}]",
                                 justify="left")
            label.grid(row=i+1, column=0, padx=5, pady=1, sticky="w")
            text_for_scaling_X.append(label)

            dropdown_var = tk.StringVar(value="Fixed value")
            dropdown_menu = ctk.CTkOptionMenu(master=X_frame,
                                              values=["Fixed value",
                                                      "Gaussian distribution",
                                                      "Truncated gaussian dist.",
                                                      "Triangular distribution",
                                                      "Uniform distribution"],
                                              variable=dropdown_var,
                                              width=180,
                                              command=partial(update_dist_params,
                                                              parent=parent,
                                                              feature=parent.X_names[i]))
            dropdown_menu.grid(row=i+1, column=1, padx=5, pady=5)

            dropdown_var_dtype = tk.StringVar(value="Real number")
            dropdown_menu_dtype = ctk.CTkOptionMenu(master=X_frame,
                                                    values=["Real number",
                                                            "Integer"],
                                                    variable=dropdown_var_dtype,
                                                    width=180,
                                                    command=partial(update_dist_params,
                                                                    parent=parent,
                                                                    feature=parent.X_names[i]))
            dropdown_menu_dtype.grid(row=i+1, column=2, padx=5, pady=5)

            box_for_scaling_X.append(dropdown_menu)

            parameter_1 = tk.StringVar(value="Value")
            textbox_1 = tk.Entry(X_frame,
                                 textvariable=parameter_1,
                                 fg="grey")
            textbox_1.bind("<FocusIn>", partial(handle_focus_in,
                                                parent=parent,
                                                feature=parent.X_names[i],
                                                box_identity=1))
            textbox_1.grid(row=i+1, column=3, padx=(10, 0), pady=5)

            parameter_2 = tk.StringVar(value="")
            textbox_2 = tk.Entry(X_frame,
                                 textvariable=parameter_2,
                                 state="disable",
                                 fg="grey")
            textbox_2.bind("<FocusIn>", partial(handle_focus_in,
                                                parent=parent,
                                                feature=parent.X_names[i],
                                                box_identity=2))
            textbox_2.grid(row=i+1, column=4, padx=(10, 0), pady=5)

            parameter_3 = tk.StringVar(value="")
            textbox_3 = tk.Entry(X_frame,
                                 textvariable=parameter_3,
                                 state="disable",
                                 fg="grey")
            textbox_3.bind("<FocusIn>", partial(handle_focus_in,
                                                parent=parent,
                                                feature=parent.X_names[i],
                                                box_identity=3))
            textbox_3.grid(row=i+1, column=5, padx=(10, 0), pady=5)

            parameter_4 = tk.StringVar(value="")
            textbox_4 = tk.Entry(X_frame,
                                 textvariable=parameter_4,
                                 state="disable",
                                 fg="grey")
            textbox_4.bind("<FocusIn>", partial(handle_focus_in,
                                                parent=parent,
                                                feature=parent.X_names[i],
                                                box_identity=4))
            textbox_4.grid(row=i+1, column=6, padx=(10, 0), pady=5)

            parent.dist_var_dict[parent.X_names[i]] = [dropdown_var, parameter_1, parameter_2,
                                                       parameter_3, parameter_4, dropdown_var_dtype]
            parent.dist_box_dict[parent.X_names[i]] = [dropdown_var, textbox_1, textbox_2,
                                                       textbox_3, textbox_4]

        parent.data_generation_window.update()
        index_longest_text_X = parent.X_names.index(
            max(parent.X_names, key=len))

        frame_width_X = text_for_scaling_X[index_longest_text_X].winfo_width() + \
            parent.dist_box_dict[parent.X_names[i]][1].winfo_width() +\
            parent.dist_box_dict[parent.X_names[i]][2].winfo_width() +\
            parent.dist_box_dict[parent.X_names[i]][3].winfo_width() +\
            2*box_for_scaling_X[index_longest_text_X].winfo_width()

        X_frame.configure(width=frame_width_X*parent.scaling_factor)

        input_header = ctk.CTkLabel(X_frame,
                                    text="Features",
                                    font=ctk.CTkFont(size=15, weight="bold"),
                                    justify="left",
                                    anchor="w")
        input_header.grid(row=0, column=0, pady=(20, 10), sticky="n")

        dist_header = ctk.CTkLabel(X_frame,
                                   text="Distribution",
                                   font=ctk.CTkFont(size=15, weight="bold"),
                                   justify="left",
                                   anchor="w")
        dist_header.grid(row=0, column=1, pady=(20, 10), sticky="n")

        dist_header = ctk.CTkLabel(X_frame,
                                   text="Number type",
                                   font=ctk.CTkFont(size=15, weight="bold"),
                                   justify="left",
                                   anchor="w")
        dist_header.grid(row=0, column=2, pady=(20, 10), sticky="n")

        param_1_header = ctk.CTkLabel(X_frame,
                                      text="Parameter 1",
                                      font=ctk.CTkFont(size=15, weight="bold"),
                                      justify="right",
                                      anchor="w")
        param_1_header.grid(row=0, column=3, pady=(20, 10), sticky="n")

        param_2_header = ctk.CTkLabel(X_frame,
                                      text="Parameter 2",
                                      font=ctk.CTkFont(size=15, weight="bold"),
                                      justify="right",
                                      anchor="w")
        param_2_header.grid(row=0, column=4, pady=(20, 10), sticky="n")

        param_3_header = ctk.CTkLabel(X_frame,
                                      text="Parameter 3",
                                      font=ctk.CTkFont(size=15, weight="bold"),
                                      justify="right",
                                      anchor="w")
        param_3_header.grid(row=0, column=5, pady=(20, 10), sticky="n")

        param_4_header = ctk.CTkLabel(X_frame,
                                      text="Parameter 4",
                                      font=ctk.CTkFont(size=15, weight="bold"),
                                      justify="right",
                                      anchor="w")
        param_4_header.grid(row=0, column=6, pady=(20, 10), sticky="n")

        sampling_frame = ctk.CTkFrame(parent.data_generation_window,
                                      height=200,
                                      width=frame_width_X*parent.scaling_factor,
                                      corner_radius=0)
        sampling_frame.grid(row=1, column=0, sticky="ns", padx=20)

        optimize_button = ctk.CTkButton(sampling_frame,
                                        width=150,
                                        text="Generate dataset",
                                        command=lambda: generate_dataset(parent))
        optimize_button.grid(row=1, column=0, padx=20, pady=(10, 10))

        n_samples_label = ctk.CTkLabel(sampling_frame,
                                       text="Enter number of samples:",
                                       font=ctk.CTkFont(
                                           size=15, weight="bold"),
                                       justify="left",
                                       anchor="w")
        n_samples_label.grid(row=1, column=1, pady=(10, 10))

        parent.n_samples_data_gen = tk.StringVar()
        textbox_n_samples = tk.Entry(sampling_frame,
                                     textvariable=parent.n_samples_data_gen)
        textbox_n_samples.grid(row=1, column=2, pady=(10, 10), padx=20)


def generate_dataset(parent):
    error = 0
    sampled_dataset = pd.DataFrame()
    sampled_dataset.index.name = "Trial values"
    if parent.n_samples_data_gen.get() == "":
        tk.messagebox.showerror("Error message", "The number of samples has not been entered.",
                                parent=parent.data_generation_window)
    else:
        n_samples = int(parent.n_samples_data_gen.get())
    for feature in parent.X_names:
        dist = parent.dist_var_dict[feature][0].get()
        try:
            if dist == "Fixed value":
                feature_vector = np.repeat(
                    float(parent.dist_var_dict[feature][1].get()), n_samples)
            elif dist == "Uniform distribution":
                if float(parent.dist_var_dict[feature][1].get()) < float(parent.dist_var_dict[feature][2].get()):
                    feature_vector = np.random.uniform(float(parent.dist_var_dict[feature][1].get()),
                                                       float(
                                                           parent.dist_var_dict[feature][2].get()),
                                                       n_samples)
                else:
                    tk.messagebox.showerror("Error message",
                                            f"The lower bound of {feature} is larger than the upper bound.",
                                            parent=parent.data_generation_window)
                    error = 1
            elif dist == "Gaussian distribution":
                feature_vector = np.random.normal(float(parent.dist_var_dict[feature][1].get()),
                                                  float(
                                                      parent.dist_var_dict[feature][2].get()),
                                                  n_samples)
            elif dist == "Truncated gaussian dist.":
                mu = float(parent.dist_var_dict[feature][1].get())
                sigma = float(parent.dist_var_dict[feature][2].get())
                lower_bound = float(parent.dist_var_dict[feature][3].get())
                upper_bound = float(parent.dist_var_dict[feature][4].get())
                if lower_bound < upper_bound:
                    truncnorm_dist = st.truncnorm((lower_bound-mu)/sigma,
                                                  (upper_bound-mu)/sigma,
                                                  loc=mu,
                                                  scale=sigma)
                    feature_vector = truncnorm_dist.rvs(n_samples)
                else:
                    tk.messagebox.showerror("Error message",
                                            f"The lower bound of {feature} is larger than the upper bound.",
                                            parent=parent.data_generation_window)
                    error = 1
            elif dist == "Triangular distribution":
                if float(parent.dist_var_dict[feature][1].get()) < float(parent.dist_var_dict[feature][3].get()):
                    feature_vector = np.random.triangular(float(parent.dist_var_dict[feature][1].get()),
                                                          float(
                                                              parent.dist_var_dict[feature][2].get()),
                                                          float(
                                                              parent.dist_var_dict[feature][3].get()),
                                                          n_samples)
                else:
                    tk.messagebox.showerror("Error message",
                                            f"The lower bound of {feature} is larger than the upper bound.",
                                            parent=parent.data_generation_window)
                    error = 1
        except ValueError:
            tk.messagebox.showerror("Error message",
                                    f"At least one required parameter for {feature} has not been entered correctly.",
                                    parent=parent.data_generation_window)
            error = 1
            break
        if parent.dist_var_dict[feature][5].get() == "Integer":
            feature_vector = np.round(feature_vector, decimals=0)
        sampled_dataset[feature] = feature_vector

    counter = 0
    batch_preds = np.empty((n_samples, len(parent.y_names)))
    for y_dir in parent.y_names:
        model_type_dir = f"{parent.model_dir}/{y_dir}"
        model_type = [filename for filename in os.listdir(
            model_type_dir) if filename.endswith(".joblib")][0]
        model = load(f"{parent.model_dir}/{y_dir}/{model_type}")

        with open(f"{parent.model_dir}/{y_dir}/X_mus", "rb") as f:
            X_mus = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_stds", "rb") as f:
            X_stds = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_mus", "rb") as f:
            y_mus = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_stds", "rb") as f:
            y_stds = pickle.load(f)

        X_norm = normalize(sampled_dataset, X_mus, X_stds)
        preds = model.predict(X_norm)
        preds = denormalize(preds, y_mus, y_stds)
        if len(preds.shape) != 1:
            batch_preds[:, counter] = preds[:, 0]
        else:
            batch_preds[:, counter] = preds
        counter += 1

    y_data = pd.DataFrame(batch_preds)
    sampled_dataset = pd.concat([y_data, sampled_dataset], axis=1)
    y_headers = [i.replace("_", " ") for i in parent.y_names]
    sampled_dataset.columns.values[:len(y_headers)] = y_headers
    sampled_dataset.index.name = "Trial values"

    cur_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M")
    if error != 1:
        save_dir = parent.model_dir.rsplit("/", 2)[0]
        sampled_dataset.to_excel(
            f"{save_dir}/sampled_dataset_{cur_datetime}.xlsx")
        parent.data_generation_window.destroy()


def handle_focus_in(_, parent, feature, box_identity):
    if parent.dist_box_dict[feature][box_identity].cget("foreground") != "black":
        parent.dist_var_dict[feature][box_identity].set("")
        parent.dist_box_dict[feature][box_identity].configure(fg="black")


def update_dist_params(dist, feature, parent):
    parent.dist_box_dict[feature][1].configure(fg="gray")
    parent.dist_box_dict[feature][2].configure(fg="gray")
    parent.dist_box_dict[feature][3].configure(fg="gray")
    parent.dist_box_dict[feature][4].configure(fg="gray")
    parent.data_generation_window.focus()
    if dist == "Fixed value":
        parent.dist_var_dict[feature][1].set("Value")
        parent.dist_var_dict[feature][2].set("")
        parent.dist_var_dict[feature][3].set("")
        parent.dist_var_dict[feature][4].set("")
        parent.dist_box_dict[feature][2].configure(state="disable")
        parent.dist_box_dict[feature][3].configure(state="disable")
        parent.dist_box_dict[feature][4].configure(state="disable")
    elif dist == "Uniform distribution":
        parent.dist_var_dict[feature][1].set("Lower bound")
        parent.dist_var_dict[feature][2].set("Upper bound")
        parent.dist_var_dict[feature][3].set("")
        parent.dist_var_dict[feature][4].set("")
        parent.dist_box_dict[feature][2].configure(state="normal")
        parent.dist_box_dict[feature][3].configure(state="disable")
        parent.dist_box_dict[feature][4].configure(state="disable")
    elif dist == "Gaussian distribution":
        parent.dist_var_dict[feature][1].set("Mean")
        parent.dist_var_dict[feature][2].set("Standard deviation")
        parent.dist_var_dict[feature][3].set("")
        parent.dist_var_dict[feature][4].set("")
        parent.dist_box_dict[feature][2].configure(state="normal")
        parent.dist_box_dict[feature][3].configure(state="disable")
        parent.dist_box_dict[feature][4].configure(state="disable")
    elif dist == "Truncated gaussian dist.":
        parent.dist_var_dict[feature][1].set("Mean")
        parent.dist_var_dict[feature][2].set("Standard deviation")
        parent.dist_var_dict[feature][3].set("Lower bound")
        parent.dist_var_dict[feature][4].set("Upper bound")
        parent.dist_box_dict[feature][2].configure(state="normal")
        parent.dist_box_dict[feature][3].configure(state="normal")
        parent.dist_box_dict[feature][4].configure(state="normal")
    elif dist == "Triangular distribution":
        parent.dist_var_dict[feature][1].set("Lower bound")
        parent.dist_var_dict[feature][2].set("Mode")
        parent.dist_var_dict[feature][3].set("Upper bound")
        parent.dist_var_dict[feature][4].set("")
        parent.dist_box_dict[feature][2].configure(state="normal")
        parent.dist_box_dict[feature][3].configure(state="normal")
        parent.dist_box_dict[feature][4].configure(state="disable")

