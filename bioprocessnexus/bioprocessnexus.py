import threading
import queue
import ctypes
import customtkinter as ctk
from performance_eval import *
from model_training import *
from interact_hist import *
from mc_subsampling import *
from data_managment import *
from prediction_making import *
from hist import *
from explanations import *
from scaling_performance import *

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.window = disclaimer(master=self)

    def launch_nexus(self):
        self.window.destroy()
        self.window = nexus(master=self)
        
class disclaimer(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master=master)
        self.protocol("WM_DELETE_WINDOW", self.master.destroy)
        self.title("BioProcessNexus - 1.0.0")
        self.disclaimer_label = ctk.CTkLabel(self,
                    text="Reminder: Using external sources without citing them goes against good "+
                    "scientific practice and is considered plagiarism. Please ensure that you cite the "+
                    "original creators of datasets, models and tools you use.",
                    font=ctk.CTkFont(size=20, weight="bold"),
                    wraplength = 540,
                    justify = "center")
        self.disclaimer_label.grid(pady=10, padx=30)
        
        self.accept_button = ctk.CTkButton(self,
                                        text="I understand",
                                        font=ctk.CTkFont(size=17, weight="bold"),
                                        command=lambda:self.master.launch_nexus())
        self.accept_button.grid(row=1, column=0)
        
class nexus(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master=master)
        
        self.protocol("WM_DELETE_WINDOW", self.master.destroy)
        self.title("BioProcessNexus - 1.0.0")
        self.var_names = ["None selected"]
        self.filename = "None"
        self.prediction_inputs = []
        self.prediction_outputs = []
        self.model_loaded = 0
        self.queue = queue.Queue()
        self.button_pad_y = 10
        try:
            self.scaling_factor = 125/ctypes.windll.shcore.GetScaleFactorForDevice(0)
        except:
            self.scaling_factor = 1.25
        # Frame 1
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=10, sticky="nsew", padx=20)
        self.sidebar_frame.grid_rowconfigure(7, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame,
                                       text="Generate model",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=self.button_pad_y)
        
        # Buttons
        self.button_width=200
        
        self.sel_file_button = ctk.CTkButton(self.sidebar_frame,
                                             width = self.button_width,
                                             text="Load data",
                                             command=lambda:set_file_dir(self))
        self.sel_file_button.grid(row=1, column=0, padx=20, pady=self.button_pad_y)
        
        self.sel_y_button = ctk.CTkButton(self.sidebar_frame,
                                          width = self.button_width,
                                          text="Select responses and features",
                                          command=lambda:choose_y(self))
        self.sel_y_button.grid(row=2, column=0, padx=20, pady=self.button_pad_y)
        
        self.train_model_button = ctk.CTkButton(self.sidebar_frame,
                                                width = self.button_width,
                                                text="Train surrogate models",
                                                command=lambda:train_models(self))
        self.train_model_button.grid(row=3, column=0, padx=20, pady=self.button_pad_y)

        self.mixture_model_button = ctk.CTkButton(self.sidebar_frame,
                                                width = self.button_width,
                                                text="Mix of experts model",
                                                command=lambda:mix_models(self))
        self.mixture_model_button.grid(row=4, column=0, padx=20, pady=self.button_pad_y) 
        
        self.zip_button = ctk.CTkButton(self.sidebar_frame,
                                        width = self.button_width,
                                        text="Compress (zip) folder",
                                        command=lambda:zip_dir(self))
        self.zip_button.grid(row=5, column=0, padx=20, pady=self.button_pad_y)
        
        self.zip_button = ctk.CTkButton(self.sidebar_frame,
                                        width = self.button_width,
                                        text="Uncompress (unzip) folder",
                                        command=lambda:unzip_dir(self))
        self.zip_button.grid(row=6, column=0, padx=20, pady=self.button_pad_y) 
        
        self.help_button = ctk.CTkButton(self.sidebar_frame,
                                         width = self.button_width,
                                         text="Help!",
                                         command=lambda:open_help())
        self.help_button.grid(row=9, column=0, padx=20, pady=self.button_pad_y)
        
        self.gr_plots_switch_var = ctk.StringVar(value="off")
        self.gr_plots_switch = ctk.CTkSwitch(self.sidebar_frame,
                                         text="Group plots",
                                         variable=self.gr_plots_switch_var)
        self.gr_plots_switch.grid(row=8, column=0, padx=20, pady=self.button_pad_y)
        
        # Frame 2
        self.sidebar_frame_2 = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame_2.grid(row=0, column=1, rowspan=10, sticky="nsew", padx=20)
        self.sidebar_frame_2.grid_rowconfigure(7, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame_2,
                                       text="Evaluate model",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=self.button_pad_y)

        # Buttons
        self.sel_model_button = ctk.CTkButton(self.sidebar_frame_2,
                                              width = self.button_width,
                                              text="Load model",
                                              command=lambda:choose_model(self))
        self.sel_model_button.grid(row=1, column=0, padx=20, pady=self.button_pad_y)
        
        self.plot_predict_button = ctk.CTkButton(self.sidebar_frame_2,
                                                width = self.button_width,
                                                text="Assess prediction performance",
                                                command=lambda:init_plot_predictions(self))
        self.plot_predict_button.grid(row=2, column=0, padx=20, pady=self.button_pad_y)   
        
        self.scaling_perf_button = ctk.CTkButton(self.sidebar_frame_2,
                                                 width = self.button_width,
                                                 text="Assess data scaling performance",
                                                 command=lambda:init_data_scaling(self))
        self.scaling_perf_button.grid(row=3, column=0, padx=20, pady=self.button_pad_y)         
              
        self.exp_button = ctk.CTkButton(self.sidebar_frame_2,
                                        width = self.button_width,
                                        text="Perform sensitivity analysis",
                                        command=lambda:threading.Thread(target=make_explanation(self)).start())
        self.exp_button.grid(row=4, column=0, padx=20, pady=self.button_pad_y)  

        self.predict_button= ctk.CTkButton(self.sidebar_frame_2,
                                            width = self.button_width,
                                            text="Make predictions",
                                            command=lambda:make_predictions(self))
        self.predict_button.grid(row=5, column=0, padx=20, pady=self.button_pad_y)   
        
        self.batch_template_button = ctk.CTkButton(self.sidebar_frame_2,
                                                  width = self.button_width,
                                                  text="Make batch predictions",
                                                  command=lambda:batch_prediction_welcome(self))
        self.batch_template_button.grid(row=6, column=0, padx=20, pady=self.button_pad_y)

        self.generate_data_button = ctk.CTkButton(self.sidebar_frame_2,
                                                  width = self.button_width,
                                                  text="Perform Monte Carlo sampling",
                                                  command=lambda:generate_data_interface(self))
        self.generate_data_button.grid(row=7, column=0, padx=20, pady=self.button_pad_y)
        
        self.plot_predict_button = ctk.CTkButton(self.sidebar_frame_2,
                                                 width = self.button_width,
                                                 text="Fit histogram",
                                                 command=lambda:init_hist(self))
        self.plot_predict_button.grid(row=8, column=0, padx=20, pady=self.button_pad_y)  
    
        self.batch_predict_button = ctk.CTkButton(self.sidebar_frame_2,
                                                  width = self.button_width,
                                                  text="Interactive histogram",
                                                  command=lambda:interactive_hist_welcome(self))
        self.batch_predict_button.grid(row=9, column=0, padx=20, pady=self.button_pad_y) 

if __name__ == "__main__":
    app = App()
    app.mainloop()
