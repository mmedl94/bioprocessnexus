import os
import shutil
from datetime import datetime
import tkinter as tk
import pickle
from functools import partial
import numpy as np
from pandas import read_csv, read_excel
import customtkinter as ctk
from helpers import *


def set_file_dir(parent):
    tk.messagebox.showinfo(
        "Information", "Plase select the file containing the data.")
    parent.model_loaded = 0
    if hasattr(parent, "y_names") is True:
        del parent.y_names
    file_types = [("CSV or Excel files", "*.csv"),
                  ("CSV or Excel files", "*.xlsx"),
                  ("CSV or Excel files", "*.xls")]
    parent.filename = ctk.filedialog.askopenfilename(filetypes=file_types)
    if parent.filename.endswith("xlsx") or parent.filename.endswith("xls"):
        parent.data = read_excel(parent.filename)
    if parent.filename.endswith("csv"):
        parent.data = read_csv(parent.filename)
    if list(parent.data)[0] != "Trial values":
        parent.data.rename(
            columns={parent.data.columns[0]: "placeholder"}, inplace=True)
        parent.var_names = list(parent.data)
    else:
        parent.var_names = list(parent.data)[1:]
    if list(parent.data)[0] == "placeholder":
        if "Trial values" not in parent.data["placeholder"].values:
            tk.messagebox.showerror("Error message", "The \"Trial values\" indicator is missing. " +
                                    "The BioProcessNexus expects \"Trial values\" to be written in " +
                                    "the fist column in the row where your the feature names are defined. " +
                                    "Please add it.")
        else:
            trial_index = parent.data[parent.data.iloc[:, 0].str.contains(
                "Trial values").fillna(False)].index[0]
            parent.data = parent.data.iloc[trial_index:]
            parent.data = parent.data.loc[:, ~parent.data.iloc[0].isna()]
            parent.data.columns = parent.data.iloc[0]
            parent.data = parent.data[1:]
            if parent.data.isnull().values.any():
                # remove rows with missing values
                tk.messagebox.showerror(
                    "Warning message", "Incomplete datapoints have been found and removed.")
                parent.data.dropna(inplace=True)

            # check for duplicate column names
            duplicates = set()
            for col in list(parent.data.columns):
                if col in duplicates:
                    tk.messagebox.showerror(
                        "Error message", "Multiple features have the same name.")
                else:
                    duplicates.add(col)

            if any(parent.data.dtypes == "object") is True:
                for col in list(parent.data):
                    if parent.data[col].dtypes == "O" and isinstance(parent.data[col].iloc[0], str):
                        parent.data[col] = parent.data[col].str.replace(
                            ',', '')
                parent.data = parent.data.astype(float)

            parent.data.to_csv(str.rsplit(parent.filename, "/", 1)[0]+"/reformatted_data.csv",
                               index=False, header=True)
            parent.var_names = list(parent.data)[1:]
    else:
        pass


def bool_switch(parent, name):
    parent.feature_selection[name].set("0")


def choose_y(parent):
    if hasattr(parent, "data") is False:
        tk.messagebox.showerror("Error message", "No data has been loaded.")
    else:
        parent.y_select_window = tk.Toplevel(parent)
        parent.y_select_window.title("Choose variables to predict")

        parent.y_select_window.grid_rowconfigure(0, weight=1)
        parent.y_select_window.grid_columnconfigure((0, 1), weight=1)
        if parent.data.columns[0] == "Trial values":
            parent.data = parent.data.iloc[:, 1:]

        parent.response_selection = {}
        parent.feature_selection = {}

        frame_response = ctk.CTkScrollableFrame(parent.y_select_window)
        frame_response.grid(row=0, column=0, padx=20, pady=0)
        frame_feature = ctk.CTkScrollableFrame(parent.y_select_window)
        frame_feature.grid(row=0, column=1, padx=20, pady=0)

        parent.response_label = ctk.CTkLabel(frame_response,
                                             text="Select responses",
                                             font=ctk.CTkFont(size=20, weight="bold"))
        parent.response_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        parent.feature_label = ctk.CTkLabel(frame_feature,
                                            text="Select features",
                                            font=ctk.CTkFont(size=20, weight="bold"))
        parent.feature_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        buttons_for_scaling = []
        row = 0
        for name in parent.var_names:
            variable = ctk.IntVar()
            variable.set("1")
            parent.feature_selection[name] = variable
            l = ctk.CTkSwitch(frame_feature, text=name, variable=variable)
            l.grid(row=row+1, column=0, padx=20, pady=0, sticky="ew")
            buttons_for_scaling.append(l)
            row += 1

        row = 0
        for name in parent.var_names:
            variable = ctk.IntVar()
            parent.response_selection[name] = variable
            l = ctk.CTkSwitch(frame_response, text=name, variable=variable,
                              command=partial(bool_switch,
                                              parent=parent,
                                              name=name))
            l.grid(row=row+1, column=0, padx=20, pady=0, sticky="ew")
            row += 1

        parent.y_select_window.update()
        index_longest_text = parent.var_names.index(
            max(parent.var_names, key=len))
        frame_width = (
            buttons_for_scaling[index_longest_text].winfo_width()-40)*parent.scaling_factor

        height1 = len(parent.var_names) * \
            buttons_for_scaling[index_longest_text].winfo_height()-40
        height2 = parent.y_select_window.winfo_screenheight()*0.7
        frame_height = np.min(
            (height1*parent.scaling_factor, height2*parent.scaling_factor))

        frame_response.configure(width=frame_width, height=frame_height)
        frame_feature.configure(width=frame_width, height=frame_height)

        accept_button = ctk.CTkButton(parent.y_select_window,
                                      text="Accept",
                                      command=lambda: fix_selection(parent),
                                      width=parent.button_width)
        accept_button.grid(row=1, column=0, columnspan=2,
                           padx=20, pady=20, sticky="ns")


def fix_selection(parent):
    parent.response_bool_vars = np.zeros(len(parent.response_selection))
    parent.feature_bool_vars = np.zeros(len(parent.feature_selection))

    counter = 0
    for i in parent.response_selection:
        if parent.response_selection[i].get() == 1:
            parent.response_bool_vars[counter] = 1
        counter += 1
    parent.response_bool_vars = np.array(parent.response_bool_vars, dtype=bool)
    parent.y_names = list(np.array(list(parent.response_selection))[
                          parent.response_bool_vars])
    parent.y_names = [i.replace(" ", "_") for i in parent.y_names]

    counter = 0
    for i in parent.feature_selection:
        if parent.feature_selection[i].get() == 1:
            parent.feature_bool_vars[counter] = 1
        counter += 1
    parent.feature_bool_vars = np.array(parent.feature_bool_vars, dtype=bool)
    # Update feature selectoin
    parent.feature_selection = np.array(list(parent.feature_selection))[
        parent.feature_bool_vars]
    if np.any(parent.response_bool_vars is True) is False:
        tk.messagebox.showerror("Error message", "No response has been selected.",
                                parent=parent.y_select_window)
    elif np.any(parent.feature_bool_vars is True) is False:
        tk.messagebox.showerror("Error message", "No feature has been selected.",
                                parent=parent.y_select_window)
    else:
        parent.y_select_window.destroy()


def mix_models(parent):
    parent.mix_dir = ctk.filedialog.askdirectory(
        title="Select model_links folder")
    parent.mix_dir = str.rsplit(parent.mix_dir, "/", 1)[0]+"/data"

    # Get responses
    try:
        parent.response_set = set()
        model_dictionary = {}
        for model_folder in os.listdir(parent.mix_dir):
            for response in os.listdir(parent.mix_dir+f"/{model_folder}"):
                response_path = parent.mix_dir+f"/{model_folder}/{response}"
                if os.path.isdir(response_path):
                    parent.response_set.add(response)
    except FileNotFoundError:
        tk.messagebox.showerror("Error message", "No models found in directory.",
                                parent=parent.y_select_window)

    for response in parent.response_set:
        temp_list = []
        for model_folder in os.listdir(parent.mix_dir):
            response_path = parent.mix_dir+f"/{model_folder}/{response}"

            if os.path.exists(response_path):
                temp_list.append(model_folder)

        model_dictionary[f"{response}"] = temp_list

    height1 = len(parent.response_set)*40-30
    height2 = parent.winfo_screenheight()*0.7

    frame_height = np.min((height1*parent.scaling_factor,
                           height2*parent.scaling_factor))

    # Spawn windows with list of responses
    parent.mix_window = tk.Toplevel(parent, height=frame_height)
    parent.mix_window.title("Generate mixture of experts model")

    parent.mix_window.grid_rowconfigure(0, weight=1)
    parent.mix_window.grid_columnconfigure((0, 1), weight=1)

    frame_response = ctk.CTkScrollableFrame(parent.mix_window)
    frame_response.grid(row=0, column=0, columnspan=3, padx=20, pady=0)

    parent.response_label = ctk.CTkLabel(frame_response,
                                         text="Responses",
                                         font=ctk.CTkFont(size=20, weight="bold"))
    parent.response_label.grid(row=0, column=0, padx=20, pady=(20, 10))

    parent.response_label = ctk.CTkLabel(frame_response,
                                         text="Model",
                                         font=ctk.CTkFont(size=20, weight="bold"))
    parent.response_label.grid(row=0, column=1, padx=20, pady=(20, 10))

    text_for_scaling = []
    dropdown_for_scaling = []
    row = 0
    parent.dropdown_selection = {}
    for response in parent.response_set:
        l = ctk.CTkLabel(frame_response, text=response, justify="left")
        l.grid(row=row+1, column=0, padx=20, pady=5, sticky="w")
        text_for_scaling.append(l)

        dropdown_var = tk.StringVar()
        parent.dropdown_selection[response] = dropdown_var
        dropdown_menu = ctk.CTkOptionMenu(master=frame_response,
                                          values=model_dictionary[f"{response}"],
                                          variable=dropdown_var)
        dropdown_menu.grid(row=row+1, column=1, padx=20, pady=5, sticky="w")
        dropdown_for_scaling.append(dropdown_menu)
        row += 1

    parent.mix_window.update_idletasks()

    index_longest_text = list(parent.response_set).index(
        max(list(parent.response_set), key=len))
    frame_width = text_for_scaling[index_longest_text].winfo_width() + \
        dropdown_menu.winfo_width()

    frame_response.configure(width=frame_width*parent.scaling_factor)
    l = ctk.CTkLabel(parent.mix_window, text="Enter model name:")
    l.grid(row=1, column=0, pady=20, padx=20)

    parent.mix_model_name = tk.StringVar()
    textbox_n_samples = tk.Entry(parent.mix_window,
                                 textvariable=parent.mix_model_name)
    textbox_n_samples.grid(row=1, column=1, pady=20, padx=20)

    accept_button = ctk.CTkButton(parent.mix_window,
                                  text="Accept",
                                  command=lambda: save_mixture_model(parent),
                                  width=parent.button_width)
    accept_button.grid(row=1, column=2, padx=20, pady=20, sticky="ns")


def save_mixture_model(parent):
    if parent.mix_model_name.get() == "":
        cur_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M")
        mix_model_name = "mixed_model_"+cur_datetime
    else:
        mix_model_name = parent.mix_model_name.get()
    link_dir = parent.mix_dir.rsplit("/", 1)[0]
    open(f"{link_dir}/model_links/{mix_model_name}.nexus", "a", encoding="utf-8")

    # Drop responses where no model has been selected
    parent.dropdown_selection = {
        key: value for key, value in parent.dropdown_selection.items() if value.get() != ""}

    # Check compatability
    initial_key = list(parent.dropdown_selection.keys())[0]
    with open(f"{parent.mix_dir}/{parent.dropdown_selection[initial_key].get()}/feature_selection.pkl", "rb") as f:
        feature_selection = pickle.load(f)
    feature_selection.sort()
    for key in list(parent.dropdown_selection.keys())[1:]:
        with open(f"{parent.mix_dir}/{parent.dropdown_selection[key].get()}/feature_selection.pkl", "rb") as f:
            feature_selection_ = pickle.load(f)
        feature_selection_.sort()
        if feature_selection != feature_selection_:
            tk.messagebox.showerror(
                "Error message", "The models must have the same features!")
            return

    # Create model folder and save feature names
    if os.path.exists(f"{parent.mix_dir}/{mix_model_name}") is False:
        os.mkdir(f"{parent.mix_dir}/{mix_model_name}")
    with open(f"{parent.mix_dir}/{mix_model_name}/feature_selection.pkl", "wb") as f:
        pickle.dump(feature_selection, f)

    # Copy respective folders into /data/mixture_model
    for key in parent.dropdown_selection.keys():
        src = f"{parent.mix_dir}/{parent.dropdown_selection[key].get()}/{key}"
        dest = f"{parent.mix_dir}/{mix_model_name}/{key}"
        shutil.copytree(src, dest)

    # Load new model
    parent.model_loaded = 1
    parent.model_dir = f"{parent.mix_dir}/{mix_model_name}"
    parent.feature_selection = feature_selection
    parent.y_names = []
    for response in os.listdir(parent.model_dir):
        response_path = f"{parent.model_dir}/{response}"
        if os.path.isdir(response_path):
            parent.y_names.append(response)
    parent.y_names.sort()
    tk.messagebox.showinfo("Information", f"Model saved at {parent.model_dir}")
    parent.mix_window.destroy()


def choose_model(parent):
    tk.messagebox.showinfo(
        "Information", "Please select .nexus file from the folder ~/model_links.")
    link_dir = ctk.filedialog.askopenfilename(
        filetypes=[("Model link", "*.nexus")])
    if not link_dir:
        tk.messagebox.showerror("Error message", "No model has been selected.")
        return
    if link_dir.split("/")[-1].endswith("nexus") is False:
        tk.messagebox.showerror(
            "Error message", "Plese select .nexus file from the folder ~/model_links.")
        return
    parent.model_loaded = 1
    model_name = link_dir.split("/")[-1].split(".")[0]
    base_dir = link_dir.rsplit("/", 2)[0]
    parent.model_dir = base_dir+"/data/"+model_name
    with open(parent.model_dir+"/feature_selection.pkl", "rb") as f:
        parent.feature_selection = pickle.load(f)
    parent.y_names = []
    for response in os.listdir(parent.model_dir):
        response_path = f"{parent.model_dir}/{response}"
        if os.path.isdir(response_path):
            parent.y_names.append(response)
    tk.messagebox.showinfo("Information",
                           f"{model_name} loaded.")

