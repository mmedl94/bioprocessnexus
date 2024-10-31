from datetime import datetime
import tkinter as tk
import threading
import os
import pickle
import numpy as np
from joblib import dump
import customtkinter as ctk
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from helpers import *


def PLS_optimize_train(X_train, y_train, splitting_ratio=5):
    # Split into train and val
    strat_indices = np.argsort(y_train)
    strat_indices_val = strat_indices[::splitting_ratio]
    strat_indices_train = np.array(list(set(strat_indices) -
                                        set(strat_indices_val)))

    X_val = X_train[strat_indices_val].copy()
    y_val = y_train[strat_indices_val].copy()

    X_train = X_train[strat_indices_train].copy()
    y_train = y_train[strat_indices_train].copy()

    performance_logger = np.inf
    for i in range(X_train.shape[1]-1):
        model = PLSRegression(n_components=i+1)
        model.fit_transform(X_train, y_train)
        y_preds = model.predict(X_val)[:, 0]
        MSE = np.mean((y_val-y_preds)**2)
        if MSE < performance_logger:
            performance_logger = MSE
            best_n_comp = i+1

    model = PLSRegression(n_components=best_n_comp)
    model.fit_transform(np.vstack((X_train, X_val)),
                        np.hstack((y_train, y_val)))
    return model


def train_models(parent):
    if hasattr(parent, "data") is False:
        tk.messagebox.showerror("Error message", "No data has been loaded.")
    elif hasattr(parent, "y_names") is False:
        tk.messagebox.showerror(
            "Error message", "Responses and features have not been selected.")
    elif not parent.y_names:
        tk.messagebox.showerror(
            "Error message", "Responses and features have not been selected.")
    else:
        parent.training_window = tk.Toplevel(parent)
        parent.training_window.title("Train model")
        parent.training_window.grid_rowconfigure((0, 1, 2, 3), weight=0)
        parent.training_window.grid_columnconfigure((0), weight=1)
        parent.close_command = tk.IntVar()

        parent.status_text_var = tk.StringVar()
        parent.status_text = ctk.CTkLabel(parent.training_window,
                                          textvariable=parent.status_text_var,
                                          font=ctk.CTkFont(size=20, weight="bold"))
        parent.status_text.grid(row=0, column=0, padx=0,
                                pady=(20, 5), columnspan=2)
        parent.status_text_var.set("Please select model")

        parent.model_name_text = ctk.CTkLabel(parent.training_window,
                                              text="Enter model name: ")
        parent.model_name_text.grid(
            row=2, column=0, padx=(20, 0), pady=(20, 5))

        parent.save_model_var = tk.StringVar()
        save_model_entry = tk.Entry(parent.training_window,
                                    textvariable=parent.save_model_var)
        save_model_entry.grid(row=2, column=1, padx=(0, 20), pady=(20, 5))
        parent.PLS_button = ctk.CTkButton(parent.training_window,
                                          width=parent.button_width,
                                          text="Train partial least squares",
                                          command=lambda: threading.Thread(target=train_PLS, args=[parent]).start())
        parent.PLS_button.grid(row=3, column=0, padx=10,
                               pady=(20, 5), columnspan=2)

        parent.RF_botton = ctk.CTkButton(parent.training_window,
                                         width=parent.button_width,
                                         text="Train random forest",
                                         command=lambda: threading.Thread(target=train_RF, args=[parent]).start())
        parent.RF_botton.grid(row=4, column=0, padx=10,
                              pady=(20, 5), columnspan=2)

        parent.GP_button = ctk.CTkButton(parent.training_window,
                                         width=parent.button_width,
                                         text="Train Gaussian process",
                                         command=lambda: threading.Thread(target=train_GP, args=[parent]).start())
        parent.GP_button.grid(row=5, column=0, padx=10,
                              pady=(20, 5), columnspan=2)

        parent.training_window.wait_variable(parent.close_command)
        parent.training_window.destroy()


def train_save_params(parent, mother_dir, model_type):
    if os.path.exists(f"{mother_dir}/model_links") is False:
        os.mkdir(f"{mother_dir}/model_links")
    if os.path.exists(f"{mother_dir}/data") is False:
        os.mkdir(f"{mother_dir}/data")
    if parent.save_model_var.get() == "":
        cur_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M")
        os.mkdir(f"{mother_dir}/data/{model_type}_{cur_datetime}")
        save_folder = f"data/{model_type}_{cur_datetime}"
        open(f"{mother_dir}/model_links/{model_type}_{cur_datetime}.nexus", "a")
    else:
        save_folder = f"data/{parent.save_model_var.get()}"
        open(f"{mother_dir}/model_links/{parent.save_model_var.get()}.nexus", "a")

    if os.path.exists(f"{mother_dir}/{save_folder}") is False:
        os.mkdir(f"{mother_dir}/{save_folder}")

    with open(f"{mother_dir}/{save_folder}/feature_selection.pkl", "wb") as f:
        pickle.dump(list(parent.feature_selection), f)
    return save_folder


def initial_normalize(array):
    mus = np.mean(array, axis=0)
    stds = np.std(array, axis=0)
    stds = stds+0.000000000000001
    normalized_array = (array-mus)/stds
    return normalized_array, mus, stds


def train_PLS(parent, splitting_ratio=5):
    parent.status_text_var.set("Please wait")
    parent.training_window.update_idletasks()
    mother_dir = str.rsplit(parent.filename, "/", 1)[0]
    model_type = "partial_least_squares"
    save_folder = train_save_params(parent, mother_dir, model_type)

    X_vars = np.array(parent.data)[:, parent.feature_bool_vars]
    y_vars = np.array(parent.data)[:, parent.response_bool_vars]

    # Normalize
    X_vars, X_mus, X_stds = initial_normalize(X_vars)
    y_vars, y_mus, y_stds = initial_normalize(y_vars)

    for i in range(y_vars.shape[1]):
        y = y_vars[:, i]
        strat_indices = np.argsort(y)
        strat_indices_test = strat_indices[::splitting_ratio]
        strat_indices_train = np.array(
            list(set(strat_indices)-set(strat_indices_test)))

        # Define training and test data
        X_test = X_vars[strat_indices_test]
        y_test = y[strat_indices_test]
        X_train = X_vars[strat_indices_train]
        y_train = y[strat_indices_train]

        pls = PLS_optimize_train(X_train, y_train)

        response_name = np.array(list(parent.response_selection))[
            np.where(parent.response_bool_vars == True)[0][i]]
        response_name = response_name.replace(" ", "_")
        response_name = response_name.replace("/", "_")

        if os.path.exists(f"{mother_dir}/{save_folder}/{response_name}") is False:
            os.mkdir(f"{mother_dir}/{save_folder}/{response_name}")

        with open(f"{mother_dir}/{save_folder}/{response_name}/X_test", "wb") as f:
            pickle.dump(X_test, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_test", "wb") as f:
            pickle.dump(y_test, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_train", "wb") as f:
            pickle.dump(X_train, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_train", "wb") as f:
            pickle.dump(y_train, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/min", "wb") as f:
            pickle.dump(np.array(X_vars.min(axis=0)), f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/max", "wb") as f:
            pickle.dump(np.array(X_vars.max(axis=0)), f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_mus", "wb") as f:
            pickle.dump(X_mus, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_stds", "wb") as f:
            pickle.dump(X_stds, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_mus", "wb") as f:
            pickle.dump(y_mus[i], f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_stds", "wb") as f:
            pickle.dump(y_stds[i], f)

        dump(
            pls, f"{mother_dir}/{save_folder}/{response_name}/{model_type}.joblib")
    parent.status_text_var.set("Done")
    parent.training_window.update_idletasks()
    parent.model_dir = f"{mother_dir}/{save_folder}"
    parent.model_loaded = 1
    model_dir = parent.model_dir.replace("data", "model_links")

    tk.messagebox.showinfo("Information",
                           f"Model link saved at {model_dir}")
    parent.close_command.set(1)


def train_GP(parent, splitting_ratio=5):
    parent.status_text_var.set("Please wait")
    parent.training_window.update_idletasks()
    mother_dir = str.rsplit(parent.filename, "/", 1)[0]
    model_type = "gaussian_process"
    save_folder = train_save_params(parent, mother_dir, model_type)

    X_vars = np.array(parent.data)[:, parent.feature_bool_vars]
    y_vars = np.array(parent.data)[:, parent.response_bool_vars]

    # Normalize
    X_vars, X_mus, X_stds = initial_normalize(X_vars)
    y_vars, y_mus, y_stds = initial_normalize(y_vars)
    for i in range(y_vars.shape[1]):
        response_name = np.array(list(parent.response_selection))[
            np.where(parent.response_bool_vars == True)[0][i]]
        response_name = response_name.replace(" ", "_")
        response_name = response_name.replace("/", "_")

        y = y_vars[:, i]
        strat_indices = np.argsort(y)
        strat_indices_test = strat_indices[::splitting_ratio]
        strat_indices_train = np.array(
            list(set(strat_indices)-set(strat_indices_test)))

        # Define training and test data
        X_test = X_vars[strat_indices_test]
        y_test = y[strat_indices_test]
        X_train = X_vars[strat_indices_train]
        y_train = y[strat_indices_train]

        kernel = WhiteKernel(noise_level_bounds=(
            1e-10, 1e5)) + ConstantKernel()*RBF()
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       random_state=0,
                                       n_restarts_optimizer=3,
                                       normalize_y=True).fit(X_train, y_train)

        if os.path.exists(f"{mother_dir}/{save_folder}/{response_name}") is False:
            os.mkdir(f"{mother_dir}/{save_folder}/{response_name}")

        with open(f"{mother_dir}/{save_folder}/{response_name}/X_test", "wb") as f:
            pickle.dump(X_test, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_test", "wb") as f:
            pickle.dump(y_test, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_train", "wb") as f:
            pickle.dump(X_train, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_train", "wb") as f:
            pickle.dump(y_train, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/min", "wb") as f:
            pickle.dump(np.array(X_vars.min(axis=0)), f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/max", "wb") as f:
            pickle.dump(np.array(X_vars.max(axis=0)), f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_mus", "wb") as f:
            pickle.dump(X_mus, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_stds", "wb") as f:
            pickle.dump(X_stds, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_mus", "wb") as f:
            pickle.dump(y_mus[i], f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_stds", "wb") as f:
            pickle.dump(y_stds[i], f)

        dump(
            gpr, f"{mother_dir}/{save_folder}/{response_name}/{model_type}.joblib")
        parent.training_window.update_idletasks()
    parent.status_text_var.set("Done")
    parent.training_window.update_idletasks()
    parent.model_dir = f"{mother_dir}/{save_folder}"
    parent.model_loaded = 1
    model_dir = parent.model_dir.replace("data", "model_links")
    tk.messagebox.showinfo("Information",
                           f"Model link saved at {model_dir}")
    parent.close_command.set(1)


def train_RF(parent, splitting_ratio=5):
    parent.status_text_var.set("Please wait")
    parent.training_window.update_idletasks()
    mother_dir = str.rsplit(parent.filename, "/", 1)[0]
    model_type = "random_forest"
    save_folder = train_save_params(parent, mother_dir, model_type)

    X_vars = np.array(parent.data)[:, parent.feature_bool_vars]
    y_vars = np.array(parent.data)[:, parent.response_bool_vars]

    # Normalize
    X_vars, X_mus, X_stds = initial_normalize(X_vars)
    y_vars, y_mus, y_stds = initial_normalize(y_vars)
    for i in range(y_vars.shape[1]):
        y = y_vars[:, i]
        strat_indices = np.argsort(y)
        strat_indices_test = strat_indices[::splitting_ratio]
        strat_indices_train = np.array(
            list(set(strat_indices)-set(strat_indices_test)))

        # Define training and test data
        X_test = X_vars[strat_indices_test]
        y_test = y[strat_indices_test]
        X_train = X_vars[strat_indices_train]
        y_train = y[strat_indices_train]

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)

        response_name = np.array(list(parent.response_selection))[
            np.where(parent.response_bool_vars == True)[0][i]]
        response_name = response_name.replace(" ", "_")
        response_name = response_name.replace("/", "_")

        if os.path.exists(f"{mother_dir}/{save_folder}/{response_name}") is False:
            os.mkdir(f"{mother_dir}/{save_folder}/{response_name}")
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_test", "wb") as f:
            pickle.dump(X_test, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_test", "wb") as f:
            pickle.dump(y_test, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_train", "wb") as f:
            pickle.dump(X_train, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_train", "wb") as f:
            pickle.dump(y_train, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/min", "wb") as f:
            pickle.dump(np.array(X_vars.min(axis=0)), f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/max", "wb") as f:
            pickle.dump(np.array(X_vars.max(axis=0)), f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_mus", "wb") as f:
            pickle.dump(X_mus, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/X_stds", "wb") as f:
            pickle.dump(X_stds, f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_mus", "wb") as f:
            pickle.dump(y_mus[i], f)
        with open(f"{mother_dir}/{save_folder}/{response_name}/y_stds", "wb") as f:
            pickle.dump(y_stds[i], f)

        dump(rf, f"{mother_dir}/{save_folder}/{response_name}/{model_type}.joblib")
        parent.training_window.update_idletasks()
    parent.status_text_var.set("Done")
    parent.training_window.update_idletasks()
    parent.model_dir = f"{mother_dir}/{save_folder}"
    parent.model_loaded = 1
    model_dir = parent.model_dir.replace("data", "model_links")
    tk.messagebox.showinfo("Information",
                           f"Model link saved at {model_dir}")
    parent.close_command.set(1)

