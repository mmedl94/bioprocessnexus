import tkinter as tk
import os
import pickle
import customtkinter as ctk
import numpy as np
from joblib import load
import pandas as pd
from pandas import read_csv, read_excel
from optimizer import *


def make_predictions(parent):
    if parent.model_loaded == 0:
        tk.messagebox.showerror("Error message", "No model has been loaded.")
    else:
        # Load variable names
        with open(f"{parent.model_dir}/feature_selection.pkl", "rb") as f:
            parent.X_names = pickle.load(f)
        parent.y_names = []
        for response in os.listdir(parent.model_dir):
            response_path = f"{parent.model_dir}/{response}"
            if os.path.isdir(response_path):
                parent.y_names.append(response)

        parent.y_dim = len(parent.y_names)
        X_dim = len(parent.X_names)

        # Set boundaries for interface
        parent.upper_boundaries_display = np.full(X_dim, np.inf)
        parent.lower_boundaries_display = np.full(X_dim, -np.inf)
        parent.upper_boundaries = np.full(X_dim, np.inf)
        parent.lower_boundaries = np.full(X_dim, -np.inf)
        parent.y_min = []
        parent.y_max = []
        for y_dir in parent.y_names:
            with open(f"{parent.model_dir}/{y_dir}/min", "rb") as f:
                mins = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/max", "rb") as f:
                maxs = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/y_test", "rb") as f:
                y_test = pickle.load(f)
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
            parent.y_min.append(y_test.min())
            parent.y_max.append(y_test.max())

        parent.prediction_window = tk.Toplevel(parent)
        parent.prediction_window.title("Make model predictions")

        height1 = X_dim*40-30
        height2 = parent.prediction_window.winfo_screenheight()*0.7

        frame_height = np.min((height1*parent.scaling_factor,
                               height2*parent.scaling_factor))

        X_frame = ctk.CTkScrollableFrame(
            parent.prediction_window, height=frame_height, corner_radius=0)
        X_frame.grid(row=0, column=0, sticky="ns", padx=20)

        y_frame = ctk.CTkScrollableFrame(
            parent.prediction_window, height=frame_height, corner_radius=0)
        y_frame.grid(row=0, column=1, sticky="ns", padx=20)

        get_results_button = ctk.CTkButton(parent.prediction_window,
                                           width=200,
                                           text="Calculate outputs",
                                           command=lambda: predict(parent))
        get_results_button.grid(
            row=1, column=1, columnspan=1, padx=20, pady=(20, 10))
        optimize_button = ctk.CTkButton(parent.prediction_window,
                                        width=200,
                                        text="Search optimal inputs",
                                        command=lambda: optimize(parent))
        optimize_button.grid(row=1, column=0, columnspan=1,
                             padx=20, pady=(20, 10))

        # Initialise/reset prediction inputs and outputs
        parent.prediction_inputs = []
        parent.prediction_outputs = []

        text_for_scaling_X = []
        switch_for_scaling_X = []
        box_for_scaling_X = []
        parent.fix_variable = {}
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

            entry_text = tk.StringVar()
            textbox = tk.Entry(X_frame, textvariable=entry_text, width=15)
            textbox.grid(row=i+1, column=1, padx=5, pady=5)
            parent.prediction_inputs.append(entry_text)
            box_for_scaling_X.append(textbox)

            variable = ctk.IntVar()
            variable.set("0")
            parent.fix_variable[parent.X_names[i]] = variable
            l = ctk.CTkSwitch(X_frame, text="", variable=variable, width=40)
            l.grid(row=i+1, column=2, padx=(10, 0), pady=5)
            switch_for_scaling_X.append(l)

        text_for_scaling_y = []
        for j in range(parent.y_dim):
            label = ctk.CTkLabel(y_frame, text=f"{parent.y_names[j]}")
            label.grid(row=j+1, column=0, padx=5, pady=5, sticky="w")

            entry_text = tk.StringVar()
            textbox = tk.Entry(y_frame, bg="grey", state="disabled",
                               textvariable=entry_text, width=15)
            textbox.grid(row=j+1, column=1, padx=5, pady=5)
            parent.prediction_outputs.append(entry_text)
            text_for_scaling_y.append(label)

        parent.prediction_window.update()
        index_longest_text_X = parent.X_names.index(
            max(parent.X_names, key=len))
        index_longest_text_y = parent.y_names.index(
            max(parent.y_names, key=len))

        frame_width_X = text_for_scaling_X[index_longest_text_X].winfo_width() + \
            switch_for_scaling_X[index_longest_text_X].winfo_width() +\
            box_for_scaling_X[index_longest_text_X].winfo_width()

        frame_width_y = text_for_scaling_y[index_longest_text_y].winfo_width() + \
            box_for_scaling_X[index_longest_text_X].winfo_width()

        X_frame.configure(width=frame_width_X*parent.scaling_factor)
        y_frame.configure(width=frame_width_y*parent.scaling_factor)

        input_header = ctk.CTkLabel(X_frame,
                                    text="Features",
                                    font=ctk.CTkFont(size=15, weight="bold"),
                                    justify="left",
                                    anchor="w")
        input_header.grid(row=0, column=0, pady=(
            20, 10), columnspan=2, sticky="n")

        fix_header = ctk.CTkLabel(X_frame,
                                  text="Fix feature  ",
                                  font=ctk.CTkFont(size=15, weight="bold"),
                                  justify="right",
                                  anchor="w")
        fix_header.grid(row=0, column=2, pady=(
            20, 10), columnspan=1, sticky="n")

        y_frame = ctk.CTkLabel(y_frame,
                               text="Responses",
                               font=ctk.CTkFont(size=15, weight="bold"),
                               justify="left",
                               anchor="w")
        y_frame.grid(row=0, column=0, pady=(20, 10), columnspan=2, sticky="n")


def predict(parent):
    try:
        X = np.array([float(i.get()) for i in parent.prediction_inputs])
    except:
        tk.messagebox.showerror("Error message", "Feature values were not entered correctly. " +
                                "Make sure values are entered for all features " +
                                "and dots are used as decimal separator.",
                                parent=parent.prediction_window)
    counter = 0
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
        X_norm = normalize(X, X_mus, X_stds)
        preds = model.predict(X_norm.reshape(1, -1))
        if len(preds.shape) != 1:
            preds = preds[:, 0]
        preds = denormalize(preds, y_mus, y_stds)
        preds = nice_round(preds[0])
        parent.prediction_outputs[counter].set(str(preds))
        counter += 1


def batch_prediction_welcome(parent):
    if parent.model_loaded == 0:
        tk.messagebox.showerror("Error message", "No model has been loaded.")
    else:
        parent.batch_preds_welc_window = tk.Toplevel(parent)
        parent.batch_preds_welc_window.title("Batch predictions")
        parent.template_button = ctk.CTkButton(parent.batch_preds_welc_window,
                                               width=parent.button_width,
                                               text="Generate template",
                                               command=lambda: batch_prediction_template(parent))
        parent.template_button.grid(row=0, column=0, padx=20, pady=5)

        parent.batch_preds_button = ctk.CTkButton(parent.batch_preds_welc_window,
                                                  width=parent.button_width,
                                                  text="Make batch predictions",
                                                  command=lambda: make_batch_predictions(parent))
        parent.batch_preds_button.grid(row=1, column=0, padx=20, pady=5)


def batch_prediction_template(parent):
    template_df = pd.DataFrame(
        columns=["Trial values"]+parent.feature_selection)
    template_df.to_excel(
        parent.model_dir+"/batch_pred_template.xlsx", index=False)
    tk.messagebox.showinfo(
        "Information", f"Template created at {parent.model_dir}")


def make_batch_predictions(parent):
    if parent.model_loaded == 0:
        tk.messagebox.showerror("Error message", "No model has been loaded.")
    else:
        filename = ctk.filedialog.askopenfilename()
        if filename.endswith("xlsx") or filename.endswith("xls"):
            batch_data = read_excel(filename)
        if filename.endswith("csv"):
            batch_data = read_csv(filename)
        if list(batch_data)[0] != "Trial values":
            batch_data.rename(
                columns={batch_data.columns[0]: "placeholder"}, inplace=True)

        if list(batch_data)[0] == "placeholder":
            if "Trial values" not in batch_data["placeholder"].values:
                tk.messagebox.showerror("Error message", "The \"Trial values\" indicator is missing. " +
                                        "The BioProcessNexus expects \"Trial values\" to be written in " +
                                        "the fist column in the row where your the feature names are defined. " +
                                        "Please add it.")
            else:
                trial_index = batch_data[batch_data.iloc[:, 0].str.contains(
                    "Trial values").fillna(False)].index[0]
                batch_data = batch_data.iloc[trial_index:]
                batch_data = batch_data.loc[:, ~batch_data.iloc[0].isna()]
                batch_data.columns = batch_data.iloc[0]
                batch_data = batch_data[1:]
                if any(batch_data.dtypes == "object") is True:
                    for col in list(batch_data):
                        if batch_data[col].dtypes == "O" and isinstance(batch_data[col].iloc[0], str):
                            batch_data[col] = batch_data[col].str.replace(
                                ',', '')
                    batch_data = batch_data.astype(float)
        else:
            pass
        # Load variable names
        with open(f"{parent.model_dir}/feature_selection.pkl", "rb") as f:
            X_names = pickle.load(f)

        y_names = []
        for response in os.listdir(parent.model_dir):
            response_path = f"{parent.model_dir}/{response}"
            if os.path.isdir(response_path):
                y_names.append(response)

        batch_data = batch_data[X_names]
        batch_preds = np.empty((batch_data.shape[0], len(y_names)))

        counter = 0
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

            X_norm = normalize(batch_data, X_mus, X_stds)
            preds = model.predict(X_norm)
            preds = denormalize(preds, y_mus, y_stds)
            if len(preds.shape) != 1:
                batch_preds[:, counter] = preds[:, 0]
            else:
                batch_preds[:, counter] = preds
            counter += 1

        batch_preds = pd.DataFrame(np.vstack((np.array(y_names), batch_preds)))
        batch_preds.columns = batch_preds.iloc[0]
        batch_preds = batch_preds[1:]
        batch_data.reset_index(inplace=True, drop=True)
        batch_preds.reset_index(inplace=True, drop=True)
        batch_preds = pd.concat([batch_preds, batch_data], axis=1)
        cur_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M")
        if filename.endswith("xlsx"):
            batch_preds.to_excel(str.rsplit(filename, "/", 1)[0]+"/batch_predictions_" +
                                 cur_datetime+".xlsx")
        if filename.endswith("csv"):
            batch_preds.to_csv(str.rsplit(filename, "/", 1)[0]+"/batch_predictions_" +
                               cur_datetime+".csv")
        tk.messagebox.showinfo(
            "Information", f"Predictions saved at {parent.model_dir}")
        parent.batch_preds_welc_window.destroy()

