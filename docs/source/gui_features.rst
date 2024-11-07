General GUI features
====================

Monte Carlo Sampling
---------------------
Here will be the part that explains how to perform the MC sampling using freeware

Formatting Monte Carlo data
---------------------------
To load the data into the BioProcessNexus, the button "Select data file" has to pressed,
which opens up a file browser with which a .csv, .xls or .xlsx file cotaining the data has to be selected.

For .csv files: The Software expects the "," as column separators and "." as decimal separators.

The string "Trial values" marks the beginning of the segment containing the data and has
to be in the upper-left corner of the data.

Take note that all values within the data file have to have a sufficient number of decimal
places otherwise the quality of the surrogate model will decrease.

Loading a model
---------------
To load a model first press the "Load model" button and then select the corresponding
.nexus-file in the ~/model_links/ subdirectory of you project folder.

Training a model
----------------
To train your own model you first have to load a dataset by pressing the
"Load data" button and then selecting a .csv, .xls or .xlsx file contraining
correctly formatted Monte-Carlo data (see "Formatting Monte Carlo data" for more information).
Then you have to select the responses and features after pressing the
"Select responses and features" button. The responses are the variables that
the model should predict and the features the varables that the model should
use to base its prediction on. Variables should never be used as response and 
eature simultaneously! Once you have selected the desired responses and features,
press the "Accept" button. Thereafter, press the "Train surrogate models" button and
select one of random forest, Gaussian process or partial least squared models. Model
training will start immediately once one of the buttons is being pressed.

Attention: the BioProcessNexus will not be responsive during model training.
Do not panic. It will be responsive again once training has finished. 

After training your model will be loaded and saved in the ~/model_links/ subdirectory.

Plotting histograms
-------------------
If you want to plot a histogram of a response, first you have to either load a
dataset first and then select responses and features, or load a model. Then a
histogram of all slected responses will be plotted and the underlying distribution of
the data be estimated. The plots will be saved in the ~/images/ subdirectory and the
details of the distributions in the ~/logs/ subdirectory. 

Attention: Be aware that the distributions of your features will influence the shape of the histograms of your responses! More information on the topic can be found here.  

The interactive histrogram
--------------------------
If you want to plot an interactive histogram of a response, first you have
to either load a dataset first and then select responses and features, or
load a model. Thereafter, press the "Interactive histogram" button. As only one
interactive histogram can be active at the time, you will have to select the specific
response you want to investigate. A window will pop up with the histogram of
the response on the left and sliders for the features on the right. By changing
the sliders of a feature you will exclude all datapoints that have values of the
corresponding feature that lie outside of the specified range. This can give insight
on how certrain features influence the shape of the histogram.

Attention: The more restrictions you apply to the data, the smaller the remaining dataset will be. At some point too few to datapoints will be left for the histogram to be useful. If you want to investigate very specific settings we recommend performing a new Monte Carlo sampling. For details see "Subsampling a new Monte-Carlo dataset".

Attention: Be aware that the distributions of your features will influence the shape of the histograms of your responses! More information on the topic can be found here.  

Assessment of a models predictive perfomrance
---------------------------------------------
It is essential to assess the accuracy of a surrogate model. A surrogate
model with a low accuracy will return different predictions for the responses
compared to the original model. As a result, all metrics generated with the
BioProcessNexus will be off. Hold-out validation with a test set size of 20% of
the original dataset is implemented in the BioProcessNexus. That means that 20% of
the samples are held back during model training. Then the model is tasked to predict
the response for these samples and the predicted value is compared with the original
value. This comparison serves as a good assessment of model accuracy as the model
wasn´t trained with these samples.

To perform this analysis with the BioProcessNexus, first load or train a model
and then click "Assess prediction performance". This will produce plots of model
predictions vs. target values for each response. Predictions that are close to the
target value will be on the diagonal line of the plot. On the contraty, predictions
that are off will be further away from the diagonal line. Additionally, the root
mean squared error (RMSE) and the normalized root mean squared error (NRMSE) are be
calculated. The RMSE is a measure of accuracy and defined as the square root of the
average squared errors. The NRMSE is a normalized version of the RMSE that can be
expressed as a percentage. Low RMSEs and NRMSEs indicte a low error and thus a high model accuracy. 

Assessment of the scaling of a models predictive performance with sample size
-----------------------------------------------------------------------------
Some model types require more samples to achieve a high prediciton performance
compared to others. Therefore, it is of importance to assess how well any given
model scales with the size of the dataset. This is done with the BioProcessNexus
by training models of the same type as the selected one, but with an artificially
reduced dataset. The model performance is then assessed on the test set. At some
point the prediction error will pleateau off and won´t become smaller anymore
with more datapoints. In that case the original dataset is of sufficient size.
However, when the perdiction error is still declining with additional datapoints,
it is suggested to perform a Monte-Carlo analysis with a larger sample size to improve model accuracy.

To assess the scaling performance of a model type with the BioProcessNexus click
"Assess data scaling performance". A new window will appear that asks for the number of
evaluations for each response. For instance, if a value 10 is entered 10 different models
will be trained on 10 datasets of different sizes. Large numbers will therefore, increase
the resolution of the analysis, but also require more computational resources.

Sensititivity analysis with SHAP
--------------------------------
Performing a sensitivity analysis gives you an idea on how much each
feature influences each response. This information is essential to decide,
which of the features are the most worthwhile of improving.

To perform a sensitvity analsys first train or load a model and click on
"Perform sensitivity analysis". You will subsequently for what fraction of
the test set the Shapley-values should be calculated. A larger fraction will
give you a more comprehensive overview on the influence of the features on the
response, however it will also take more computation resources. This will produce
a beeswarm-plot of the Shapley values for each response. 

Making model predictions
------------------------
If you want to use a model to predict the responses, you first have
to train or load a model and then click "Make predictions". This will
open a interface to make model predictions. To do so you have to enter
all feature values and then press "Calculate outputs", which will result
in the respective values for the responses being displayed under "Responses". 

Attention: Using feature values that are outside of the displayed boundaries
will result in extrapolation of the model. Therefore, going outside of the
boundaries is NOT reccomended. The performance metrics for a model only hold
true during interpolation and not during extrapolation. Additionally, the prediction
performance is expected to decline the further outside of the boundaries a datapoint lies.

Finding the optimal feature values
----------------------------------
If you want to optimize the features for a model,
you first have to load or train that model and then click on
"Make predictions". This will open an interface that model predictions
can be made with. In that interface click on "Search optimal inputs". Before
the optimzer is started a few parameters have to be defined.

The number of iterations of the optimizer: How many iterations should be
performed by the optimizer. A too small number of iterations will result
in a quick result, but the found optimium might be suboptimal. A too large
number will take unnecessarily long, but will result in a better solution.
How many iterations the optimizer requires is dependant on the complexity
of the problem.

The weights for the responses: Each response has a corresponding weight.
If a response has a large weight, the optimizer will prioritize optimizing
that response. A weight of 0 results in the optimizer not taking that response
into account at all. The importance of the weights are relative to each
other; e.g. when all responses have the same weight, the absolute value
of that weight doesn´t matter.

Whether the optimizer should maximize or minimize the features for a response.

Optional: If you cannot influence a variable (e.g. the process yield)
and therefore don´t want it to be taken into account during optimization,
you can click the switch in the "Fix feature" column of the prediction interface. 
Once all parameters have been set click "Accept" and the optimizer will start.
This can take some time.

Once the optimizer has finished, the found feature values will be entered
in the model prediction interface and saved under ~/logs.

Making predictions for multiple datapoints at the same time
-----------------------------------------------------------
As making multiple predictions can be a lot of work with the
prediction interface, we also implemented the option to define
all the feature values in an .xlsx or .csv file and let the model
make it´s prediction base on that. This file must have "Trial values"
written in the first column of the first row. The first row should then
include all feature names, which must be written exactly as in the original
Monte-Carlo file. The BioProcessNexus will then read out all the data
and produce a file with the corresponding predictions. 

Therefore, you have to train or load a model and then click on
"Make batch predictions". This will open a file browser where you 
have to select the file containing the feature values. Then the predictions
will be made and saved as "batch_predictions_mm_dd_yyyy_hh_mm.csv".

Subsampling a new Monte-Carlo dataset
-------------------------------------
If you want to generate a new Monte-Carlo sampling you can
use the function "Perform Monte-Carlo sampling". This is highly 
recommended when you want to use histograms for data interpretation.
The reason for this is that original Monte-Carlo samplings were performed
with uniform distributions for the features, which good for model training,
but uniform distributions often aren´t the real underlying distributions of
the features. Using the wrong distributions will distort the histogram and
as a consequence reduce the accuracy of the histogram. 

To perform a custom Monto-Carlo sampling you will first have to load a
model and then click on "Perform Monte Carlo sampling". This will open an
interface where you will have to define the sampling distributions for all features.
Additionally, you will have to define whether the sampled values should be
real numbers (e.g. buffer cost of 5.35 €/kg) or integers (e.g. number of batches of 35).
Finally, you will have to enter how many samples should be taken to the textbox
next to "Enter number of samples:" and click "Generate dataset". The new Monte-Carlo
dataset will then be saved as "sampled_dataset_mm_dd_yyyy_hh_mm.xlsx". 

Attention: Using feature values that are outside of the displayed boundaries will
result in extrapolation of the model. Therefore, going outside of the boundaries 
is NOT reccomended. The performance metrics for a model only hold true during
interpolation and not during extrapolation. Additionally, the prediction performance
is expected to decline the further outside of the boundaries a datapoint lies.