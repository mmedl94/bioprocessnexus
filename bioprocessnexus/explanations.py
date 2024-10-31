import os
import threading
import tkinter as tk
from functools import partial
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from joblib import load
from PIL import Image, ImageTk
import shap
from shap.plots.colors._colors import red_blue
import customtkinter as ctk
from helpers import *


def make_explanation(parent):
    if parent.model_loaded == 0:
        tk.messagebox.showerror("Error message", "No model has been loaded.")
    else:
        parent.exp_window = tk.Toplevel(parent)
        parent.exp_window.title("SHapley Additive exPlanations")

        parent.exp_label_text = tk.StringVar()
        parent.exp_label = ctk.CTkLabel(parent.exp_window,
                                        textvariable=parent.exp_label_text,
                                        font=ctk.CTkFont(
                                            size=15, weight="bold"),
                                        wraplength=1000)
        parent.exp_label_text.set("Please enter fraction of samples \u00B9/\u2093 to be analyzed\n" +
                                  "x=1   --> \u00B9/\u2081  --> 100% of samples\n" +
                                  "x=5   --> \u00B9/\u2085  --> 20%   of samples\n" +
                                  "x=20 --> \u00B9/\u2082\u2080 --> 5%     of samples")
        parent.exp_label.grid(
            row=0, column=0, columnspan=4, padx=20, pady=(20, 10))

        parent.textbox_label = ctk.CTkLabel(parent.exp_window,
                                            text="x=",
                                            font=ctk.CTkFont(size=15, weight="bold"))
        parent.textbox_label.grid(row=1, column=1, sticky="e")

        parent.fraction = tk.StringVar()
        textbox = tk.Entry(parent.exp_window, textvariable=parent.fraction)
        textbox.grid(row=1, column=2, padx=0, pady=5, sticky="w")
        parent.exp_button = ctk.CTkButton(parent.exp_window,
                                          width=parent.button_width,
                                          text="Plot explanations",
                                          command=lambda: threading.Thread(target=plot_explanation(parent)).start())
        parent.exp_button.grid(
            row=2, column=0, columnspan=4, padx=0, pady=(20, 10))


def plot_explanation(parent):
    if parent.fraction.get() == "":
        tk.messagebox.showerror("Error message", "The fraction hasn´t been specified.",
                                parent=parent.exp_window)
    else:
        parent.references = []
        parent.exp_label_text.set(
            "      This may take some time - Go get a coffe       \n\n\n")
        parent.exp_window.update_idletasks()

        # Setup grouped plot
        if parent.gr_plots_switch_var.get() == "1":
            n_subplots = len(parent.y_names)
            subplots_per_row = 3
            n_rows = (n_subplots + subplots_per_row - 1) // subplots_per_row
            if n_subplots > 3:
                fig, axs = plt.subplots(
                    n_rows, subplots_per_row, figsize=(50, 4))
            else:
                fig, axs = plt.subplots(
                    n_rows, subplots_per_row, figsize=(50, 1))
            axs = axs.flatten()
            for i in range(n_subplots, len(axs)):
                fig.delaxes(axs[i])

        shap_summary = []
        shap_summary_norm = []
        for y_dir in parent.y_names:
            pretty_y = y_dir.replace("_", " ")
            if parent.gr_plots_switch_var.get() != "1":
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_title(f"{pretty_y}")
            # Load model
            model_type_dir = f"{parent.model_dir}/{y_dir}"
            model_type = [filename for filename in os.listdir(
                model_type_dir) if filename.endswith(".joblib")][0]
            parent.model = load(f"{parent.model_dir}/{y_dir}/{model_type}")

            with open(f"{parent.model_dir}/feature_selection.pkl", "rb") as f:
                feature_selection = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/y_mus", "rb") as f:
                parent.y_mus = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/y_stds", "rb") as f:
                parent.y_stds = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/X_mus", "rb") as f:
                X_mus = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/X_stds", "rb") as f:
                X_stds = pickle.load(f)
            with open(f"{parent.model_dir}/{y_dir}/X_test", "rb") as f:
                X_test = pickle.load(f)

            parent.X_names = feature_selection
            fract = int(parent.fraction.get())
            explainer = shap.KernelExplainer(model=partial(denormalized_prediction, parent=parent),
                                             data=X_test[::fract],
                                             link="identity")
            try:
                shap_values = explainer.shap_values(
                    X=X_test[::fract], nsamples=100)
                shap_summary.append(shap_values)
                shap_summary_norm.append(
                    (shap_values-shap_values.min()) /
                    (shap_values.max()-shap_values.min()))
            except MemoryError as err_msg:
                if err_msg.__class__.__name__ == "MemoryError":
                    tk.messagebox.showerror(
                        "Error message", "Please reduce number of samples!\n" + str(err_msg))
                else:
                    tk.messagebox.showerror("Error message", str(err_msg))
                parent.exp_window.destroy()
                return

            if parent.gr_plots_switch_var.get() != "1":
                parent.exp_vis_window = tk.Toplevel(parent)
                parent.exp_vis_window.title("SHapley Additive exPlanations")
                parent.explanation_frame = ctk.CTkFrame(
                    parent.exp_vis_window, corner_radius=0)
                parent.explanation_frame.grid(
                    row=2, column=1, sticky="nsew", padx=20)
                fig = plt.gcf()
                shap.summary_plot(shap_values=shap_values,
                                  features=(X_test[::fract]+X_mus)*X_stds,
                                  feature_names=parent.X_names,
                                  plot_size=(13, 7),
                                  show=False)
                img_dir = check_dir(parent, y_dir, "images")
                fig.savefig(f"{img_dir}/shapley_values.png")
                parent.image = ImageTk.PhotoImage(
                    Image.open(f"{img_dir}/shapley_values.png"))
                img = tk.Label(parent.explanation_frame, image=parent.image)
                img.grid(row=0, column=0)
                parent.references.append(parent.image)

        if parent.gr_plots_switch_var.get() == "1":
            variances_list = [np.var(arr, axis=0) for arr in shap_summary_norm]
            # Get the top indices by variance for each array
            top_indices = [np.argsort(
                variances)[-15//n_subplots:][::-1] for variances in variances_list]
            # Combine the top indices
            combined_indices = []
            for indices in top_indices:
                combined_indices.extend(indices)
            # Remove duplicates and keep the order
            unique_indices = []
            seen = set()
            for idx in combined_indices:
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
            # If the number of unique indices is less than 15, fill with the highest average variances
            if len(unique_indices) < 15:
                average_variances = np.mean(variances_list, axis=0)
                sorted_avg_var_indices = np.argsort(average_variances)[::-1]
                for idx in sorted_avg_var_indices:
                    if idx not in unique_indices:
                        unique_indices.append(idx)
                    if len(unique_indices) == 15:
                        break

            sorted_feature_names = [parent.X_names[i] for i in unique_indices]
            subplot_counter = 0
            for y_dir in parent.y_names:
                # Load model
                model_type_dir = f"{parent.model_dir}/{y_dir}"
                model_type = [filename for filename in os.listdir(
                    model_type_dir) if filename.endswith(".joblib")][0]
                parent.model = load(f"{parent.model_dir}/{y_dir}/{model_type}")

                with open(f"{parent.model_dir}/feature_selection.pkl", "rb") as f:
                    feature_selection = pickle.load(f)
                with open(f"{parent.model_dir}/{y_dir}/y_mus", "rb") as f:
                    parent.y_mus = pickle.load(f)
                with open(f"{parent.model_dir}/{y_dir}/y_stds", "rb") as f:
                    parent.y_stds = pickle.load(f)
                with open(f"{parent.model_dir}/{y_dir}/X_mus", "rb") as f:
                    X_mus = pickle.load(f)
                with open(f"{parent.model_dir}/{y_dir}/X_stds", "rb") as f:
                    X_stds = pickle.load(f)
                with open(f"{parent.model_dir}/{y_dir}/X_test", "rb") as f:
                    X_test = pickle.load(f)

                parent.X_names = feature_selection
                fract = int(parent.fraction.get())
                pretty_y = y_dir.replace("_", " ")
                plt.sca(axs[subplot_counter])
                axs[subplot_counter].set_title(f"{pretty_y}", fontsize=8)
                shap.summary_plot(shap_values=shap_summary[subplot_counter][:, unique_indices],
                                  features=(
                                      (X_test[::fract]+X_mus)*X_stds)[:, unique_indices],
                                  feature_names=sorted_feature_names,
                                  sort=False,
                                  show=False,
                                  color_bar=False,
                                  max_display=15)
                axs[subplot_counter].tick_params(
                    axis="x", which='major', labelsize=8)
                if subplot_counter % 3 != 0:
                    axs[subplot_counter].set_yticklabels([])
                else:
                    axs[subplot_counter].tick_params(
                        axis="y", which='major', labelsize=7)
                if (subplot_counter-1) % 3 != 0:
                    axs[subplot_counter].set_xlabel("")
                else:
                    axs[subplot_counter].set_xlabel("SHAP value", fontsize=8)
                if subplot_counter-2 % 3 == 0:
                    axs[subplot_counter].set_ylabel("")
                subplot_counter += 1
            img_dir = check_dir(parent, y_dir, "images")

            fig.subplots_adjust(right=1)
            cbar_ax = fig.add_axes([1.05, 0.05, 0.02, 0.9])
            fig.colorbar(cm.ScalarMappable(norm=None, cmap=red_blue),
                         cax=cbar_ax,
                         ticks=[0, 1],
                         label="Feature value")
            cbar_ax.set_yticklabels(["Low", "High"], fontsize=8)
            fig.tight_layout()
            img_dir = parent.model_dir.replace("data", "images")
            fig.savefig(f"{img_dir}/shapley_values.png",
                        dpi=600, bbox_inches='tight')
            parent.exp_vis_window = tk.Toplevel(parent)
            parent.exp_vis_window.grid_rowconfigure(0, weight=1)
            parent.exp_vis_window.grid_columnconfigure(0, weight=1)
            parent.exp_vis_window.title("SHapley Additive exPlanations")
            canvas = FigureCanvasTkAgg(fig, master=parent.exp_vis_window)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0)
        else:
            img_dir = parent.model_dir.replace("data", "images")
        tk.messagebox.showinfo("Information", f"Images saved at {img_dir}")
        parent.exp_window.destroy()


def denormalized_prediction(data_in, parent):
    preds = parent.model.predict(data_in)
    # Vectorize predictions in case they are not
    if len(preds.shape) != 1:
        preds = preds[:, 0]
    data_out = denormalize(preds, parent.y_mus, parent.y_stds)
    return data_out

