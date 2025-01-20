import os
import threading
import tkinter as tk
import pickle
from datetime import datetime
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, space_eval
from joblib import load
import customtkinter as ctk
from .helpers import *


def target_function(param_space):
    """
    Defines the target function for optimization based on model predictions and normalized output.

    Args:
        param_space: Dictionary containing feature values, model directory, optimization ratios, and bounds.

    Retruns:
        target: Target value to minimize during optimization

    This function evaluates models based on normalized predictions, applies optimization ratios, 
    and returns a target value to minimize during optimization.
    """
    # Load X
    X = []
    for X_name in param_space["X_names"]:
        X.append(param_space[X_name])

    X = np.array(X)
    preds = []
    model_dir = param_space["model_dir"]
    # Load models
    y_names = []
    responses = os.listdir(model_dir)
    responses.remove("feature_selection.pkl")
    for response in responses:
        response_path = f"{model_dir}/{response}"
        if os.path.isdir(response_path):
            y_names.append(response)

    for y_dir in y_names:
        model_type_dir = f"{model_dir}/{y_dir}"
        model_type = [filename for filename in os.listdir(
            model_type_dir) if filename.endswith(".joblib")][0]
        model = load(f"{model_dir}/{y_dir}/{model_type}")
        pred = model.predict(X.reshape(1, -1))
        # Reshape in case the model output is not a 1D array
        if len(pred.shape) != 1:
            pred = pred[:, 0]
        preds.append(pred)

    # Calculate target
    target = (np.array(param_space["opt_ratios"])
              * np.concatenate(preds)).sum()
    return -target


def run_hopt(param_space, queue):
    """
    Runs the hyperparameter optimization using TPE and stores results in a queue.

    Args:
        param_space: Dictionary containing search space and optimization settings.
        queue: Queue object to hold the results for later retrieval.

    The function uses "fmin" to perform optimization, evaluating the target_function.
    """
    param_space_copy = {}
    for key in param_space.keys():
        param_space_copy[key] = param_space[key]
    param_space_copy["opt_ratios"] = [
        float(i.get()) for i in param_space["opt_ratios"]]
    param_space_copy["n_iter"] = int(param_space["n_iter"].get())
    # Refractor opt ratios
    for i in range(len(param_space["opt_ratios"])):
        if param_space["min_max"][i].get() == "Minimize":
            param_space_copy["opt_ratios"][i] = param_space_copy["opt_ratios"][i]*-1
    # results = Trials()
    hbo = fmin(fn=target_function,
               space=param_space_copy,
               algo=tpe.suggest,
               max_evals=param_space_copy["n_iter"],
               # trials=results,
               points_to_evaluate=param_space_copy["eval_points"])
    queue.put(hbo)


def optimize(parent):
    """
    Sets up and displays the optimization interface, allowing users to specify parameters.

    Args:
        parent: The main application instance

    Creates sliders and entry fields for setting weights and iteration limits for the optimization,
    and triggers optimization when all inputs are correctly configured.
    """
    parent.optimization_window = tk.Toplevel(parent)
    parent.optimization_window.grid_rowconfigure(0, weight=1)
    parent.optimization_window.grid_columnconfigure(0, weight=1)
    parent.optimization_window.title("Optimize inputs")

    parent.opt_label_text = tk.StringVar()
    parent.optimization_label = ctk.CTkLabel(parent.optimization_window,
                                             textvariable=parent.opt_label_text,
                                             font=ctk.CTkFont(size=15, weight="bold"))
    parent.opt_label_text.set("Which parameter do you want to optimize? " +
                              "\nPlease provide weights. 0=not important, 10=very important")
    parent.optimization_label.grid(
        row=0, column=0, columnspan=4, padx=20, pady=(20, 10))

    label = ctk.CTkLabel(parent.optimization_window,
                         text="Number of iterations of optimizer \nShould be >200",
                         justify="left")
    label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    n_iter = tk.StringVar()
    textbox = tk.Entry(parent.optimization_window,
                       textvariable=n_iter, width=5)
    textbox.grid(row=1, column=1, padx=5, pady=5)

    parent.min_max = []
    parent.optimization_ratios = []

    responses = os.listdir(parent.model_dir)
    responses.remove("feature_selection.pkl")
    for j in range(parent.y_dim):
        label = ctk.CTkLabel(parent.optimization_window,
                             text=f"{responses[j]}", justify="left")
        label.grid(row=j+2, column=0, padx=5, pady=5, sticky="w")

        entry_text = tk.DoubleVar()
        entry_text.set(5)
        textbox = tk.Entry(parent.optimization_window,
                           textvariable=entry_text, width=5)
        textbox.grid(row=j+2, column=1, padx=5, pady=5)
        parent.optimization_ratios.append(entry_text)

        slider = ctk.CTkSlider(parent.optimization_window, from_=0, to=10,
                               variable=entry_text)
        slider.grid(row=j+2, column=2, padx=5, pady=5, sticky="w")

        dropdown_var = tk.StringVar(value="Maximise")
        dropdown_menu = ctk.CTkOptionMenu(master=parent.optimization_window,
                                          values=["Maximise", "Minimize"],
                                          variable=dropdown_var)
        dropdown_menu.grid(row=j+2, column=3, padx=5, pady=5)
        parent.min_max.append(dropdown_var)

    y_dir = parent.y_names[0]
    with open(f"{parent.model_dir}/{y_dir}/X_mus", "rb") as f:
        X_mus = pickle.load(f)
    with open(f"{parent.model_dir}/{y_dir}/X_stds", "rb") as f:
        X_stds = pickle.load(f)

    param_space = {}

    n_obs_extremes = 10
    # n_obs_extremes best + n_obs_extremes worst for each y_dir
    eval_points = np.empty(
        (len(parent.y_names)*n_obs_extremes*2, X_mus.shape[0]))

    counter = 0
    for y_dir in parent.y_names:
        with open(f"{parent.model_dir}/{y_dir}/X_test", "rb") as f:
            X_test = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_test", "rb") as f:
            y_test = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_train", "rb") as f:
            X_train = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/y_train", "rb") as f:
            y_train = pickle.load(f)

        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))

        # Get indices of the n_obs_extremes smallest and n_obs_extremes largest y values
        sorted_indices = np.argsort(y)
        best_indices = sorted_indices[:n_obs_extremes]
        worst_indices = sorted_indices[-n_obs_extremes:]

        # Add the n_obs_extremes best and n_obs_extremes worst points to eval_points
        eval_points[counter:counter + n_obs_extremes] = X[best_indices]
        counter += n_obs_extremes
        eval_points[counter:counter + n_obs_extremes] = X[worst_indices]
        counter += n_obs_extremes

    for i in range(X_mus.shape[0]):
        # if the parameter is fixed use the fixed value else use an uniform distribution with the feature space
        # boundaries
        if parent.fix_variable[parent.X_names[i]].get():
            fixed_value = (
                float(parent.prediction_inputs[i].get())-X_mus[i])/X_stds[i]
            param_space[parent.X_names[i]] = hp.choice(
                parent.X_names[i], [fixed_value])
            eval_points[:, i] = fixed_value
        else:
            param_space[parent.X_names[i]] = hp.uniform(parent.X_names[i],
                                                        parent.lower_boundaries[i] -
                                                        0.0000000001,
                                                        parent.upper_boundaries[i]+0.0000000001)

    # Transfrom eval_points to dicts for hyperopt
    eval_points_dicts = [dict(zip(parent.X_names, point))
                         for point in eval_points]

    param_space["n_iter"] = n_iter
    param_space["X_dims"] = len(parent.X_names)
    param_space["X_names"] = parent.X_names
    param_space["model_dir"] = parent.model_dir
    param_space["mins"] = parent.y_min
    param_space["maxs"] = parent.y_max
    param_space["opt_ratios"] = parent.optimization_ratios
    param_space["min_max"] = parent.min_max
    param_space["eval_points"] = eval_points_dicts

    optimize_button = ctk.CTkButton(parent.optimization_window,
                                    width=200,
                                    text="Accept",
                                    command=lambda: intiate_optimizer_main(parent, param_space))
    optimize_button.grid(row=parent.y_dim+2, column=0,
                         columnspan=4, padx=20, pady=(20, 10))


def intiate_optimizer_main(parent, param_space):
    """
    Checks if all optimization weights and iteration settings are valid before starting optimization.

    Args:
        parent: The main application instance
        param_space: Dictionary of parameters to be passed to the optimization process.

    Displays errors if weights are missing, out of bounds, or if iterations are not set.
    Otherwise, begins the optimization process and provides feedback to the user.
    """
    weight_missing = True in (
        i.get() is "" for i in parent.optimization_ratios)
    if weight_missing is False:
        weight_oob = True in (float(i.get()) < 0 or float(
            i.get()) > 10 for i in parent.optimization_ratios)
    if param_space["n_iter"].get() is "":
        tk.messagebox.showerror("Error message", "Please enter number of iterations of optimizer",
                                parent=parent.optimization_window)
    elif weight_missing is True:
        tk.messagebox.showerror("Error message", "Please enter weights for all responses",
                                parent=parent.optimization_window)
    elif weight_oob is True:
        tk.messagebox.showerror("Error message", "The weights must be between 0 and 10",
                                parent=parent.optimization_window)
    else:
        parent.opt_label_text.set("This may take some time\nGo get a coffe")
        parent.optimization_window.update_idletasks()
        # Start Bayesian optimization
        parent.param_space = param_space
        hoptr(parent)


def hoptr(parent):
    """
    Starts a new thread for the hyperparameter optimization (HBO) and monitors the queue.

    Args:
        parent: The main application instance where results will be displayed.

    This function initiates a separate thread to perform optimization while monitoring the 
    queue for completed results, allowing for non-blocking UI updates.
    """
    check_hbo_queue(parent)
    threading.Thread(target=run_hopt, args=[
                     parent.param_space, parent.queue]).start()


def check_hbo_queue(parent):
    """
    Checks the HBO queue for completed optimization results and updates the interface.

    Args:
        parent: The main application instance containing the optimization settings and inputs.

    When results are found in the queue, this function applies the optimized parameters to 
    the relevant UI fields and saves a log of the optimization process.
    """
    if not parent.queue.empty():
        parent.results = parent.queue.get()

        y_dir = parent.y_names[0]
        with open(f"{parent.model_dir}/{y_dir}/X_mus", "rb") as f:
            X_mus = pickle.load(f)
        with open(f"{parent.model_dir}/{y_dir}/X_stds", "rb") as f:
            X_stds = pickle.load(f)
        for i in range(len(parent.X_names)):
            optimized_value = parent.results[parent.X_names[i]
                                             ]*X_stds[i]+X_mus[i]
            parent.prediction_inputs[i].set(nice_round(optimized_value))

        mother_dir = parent.model_dir.replace("/data/", "/logs/")
        check_dir(parent, y_dir, "logs", central_log=1)
        if os.path.exists(f"{mother_dir}/optimization_logs") is False:
            os.mkdir(f"{mother_dir}/optimization_logs")
        cur_date = datetime.now().strftime("%m_%d_%Y_%H_%M")
        save_dir = f"{mother_dir}/optimization_logs/{cur_date}.txt"
        with open(save_dir, "w", encoding="utf-8") as f:
            for i in range(len(parent.X_names)):
                f.write(
                    f"{parent.X_names[i]}: {parent.results[parent.X_names[i]]*X_stds[i]+X_mus[i]}\n")
        parent.optimization_window.destroy()
        dir = parent.model_dir.replace("data", "logs")
        tk.messagebox.showinfo("Information",
                               f"Optimal feature values saved at {dir}/optimization_logs/")
    else:
        parent.after(50, check_hbo_queue, parent)
