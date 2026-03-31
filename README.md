# DATA 612 Project Proposal
## Probabilistic Electricity Load Forecasting using Transformer-Based Time Series Models
### Aayush Verma, Kanishk Kaul, Rishi Koushik Sridharan, Samarth Singh
### 1.	Background
Electricity demand forecasting is a critical task for power grid operators and energy providers. Accurate load forecasting enables efficient energy generation scheduling, grid stability, and better integration of renewable energy sources. Traditional time-series forecasting methods such as ARIMA, exponential smoothing, and regression-based models have been widely used for electricity demand prediction. However, these methods often struggle to capture complex temporal dependencies and long-term patterns present in electricity consumption data.
Recent advances in deep learning have shown that Transformer neural networks are highly effective for sequence modeling tasks. Originally introduced for natural language processing, Transformers rely on self-attention mechanisms that allow the model to capture long-range dependencies across time steps without relying on recurrent structures. This property makes them particularly suitable for time series forecasting problems, where future electricity demand may depend on usage patterns observed over long historical windows.
In this project, we propose to apply a Transformer-based time series forecasting model to predict electricity consumption patterns using the Electricity Load Diagrams 2011–2014 dataset. The main focus of the project will be probabilistic forecasting, where the model predicts a range of likely future demand values rather than only a single point estimate. This will be achieved through quantile-based forecasting, allowing the model to generate prediction intervals that better represent uncertainty in electricity demand.
________________________________________
### 2.	Significance and Problem Statement
Electricity load forecasting plays an important role in the operation and planning of modern power systems. Grid operators must anticipate future electricity demand in order to allocate generation resources efficiently, maintain grid reliability, reduce energy waste, and avoid overproduction or shortages of electricity.
Electricity consumption exhibits strong temporal patterns such as hourly cycles, daily variations, weekly trends, and seasonal effects. At the same time, electricity demand is inherently uncertain because it is influenced by changing weather conditions, human activity, holidays, and other unpredictable factors. As a result, models that only produce point predictions may not provide enough information for real-world decision-making.
This project aims to investigate how Transformer-based deep learning models can be used for probabilistic electricity load forecasting by capturing long-range temporal dependencies and generating prediction intervals for future demand. The model will be trained on historical electricity consumption data and evaluated against baseline forecasting methods. In addition, if time permits, the project will include a comparison with a second Transformer-based architecture to study whether a different design improves forecasting accuracy or uncertainty estimation.
________________________________________

### 3.	Dataset and Data Acquisition
The dataset used for this project is the Electricity Load Diagrams 2011–2014 dataset, available from the UCI Machine Learning Repository.
Dataset source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
The dataset contains electricity consumption data for 370 customers measured every 15 minutes over a period of four years (2011–2014). Each column corresponds to the electricity consumption of a specific customer, and each row represents a timestamped observation.
For this project, the data will be preprocessed to create training and testing datasets suitable for probabilistic forecasting. To keep the scope manageable within the project timeline, the customer-level load series will be aggregated into a single hourly electricity demand series. This allows the team to focus on the forecasting methodology and quality while reducing the complexity of large-scale multivariate modeling and preprocessing.
________________________________________
### 4.	Deep Learning Framework and Model Implementation
The primary forecasting model will be implemented using the PyTorch or TensorFlow deep learning framework. The model will be designed for probabilistic multi-step forecasting, where historical electricity demand over a fixed input window is used to predict multiple quantiles of future demand over a specified forecast horizon.
The Transformer architecture will include:

•	an input feature embedding layer

•	positional encoding to represent time order

•	multiple self-attention encoder layers

•	feed-forward layers for sequence representation

•	an output layer that predicts multiple quantiles for each future time step

The input to the model will consist of historical electricity load values along with time-based covariates such as hour of day, day of week, and month. The model will be developed for multi-step forecasting, using a fixed historical input sequence to generate probabilistic forecasts for the next 24 hours. The output will be a set of quantile forecasts, such as the 10th, 50th, and 90th percentiles, for each step in the prediction horizon. This formulation will allow the model to estimate both the central tendency of future electricity demand and the associated uncertainty intervals. Training will be performed using quantile loss (pinball loss), which is a standard and widely used approach for probabilistic forecasting.

As an optional extension, if time permits, the project will implement one additional Transformer-based forecasting architecture with a different but still manageable design, such as Informer or another lightweight long-sequence forecasting model. This extension will be included only if the core forecasting pipeline is completed successfully. 

________________________________________


### 5.	Validation and Evaluation Metrics
To evaluate the performance of the forecasting models, both point forecast and probabilistic forecast metrics will be considered. Since the project focuses on probabilistic electricity load forecasting, greater emphasis will be placed on metrics that assess uncertainty estimation and interval quality. However, the final choice of evaluation metrics may be refined depending on the specific forecasting setup, data behavior, and experimental results observed during implementation.
At present, the following metrics are proposed:
Point Forecast Metrics:

•	Mean Absolute Error (MAE)

•	Root Mean Squared Error (RMSE)

•	Mean Absolute Percentage Error (MAPE)

For point forecast comparison, the median predicted value from the probabilistic model will be treated as the central forecast.
Probabilistic Forecast Metrics:

•	Quantile Loss (Pinball Loss)

•	Prediction Interval Coverage Probability (PICP)

Additional evaluation measures may also be considered if needed, depending on how well they reflect the forecasting objectives and the characteristics of the predicted uncertainty intervals.

This set of metrics is intended to support comparison of the baseline models and the Transformer-based forecasting model in terms of both predictive accuracy and quality of uncertainty estimation.
________________________________________
### 6.	Expected Outcome
The expected outcome of this project is a Transformer-based probabilistic forecasting system capable of predicting future electricity demand together with uncertainty bounds. The project is expected to demonstrate that Transformer-based models can effectively capture temporal patterns in electricity load data and provide more informative forecasts than standard point prediction approaches.

If the optional extension is completed, the project will also provide a comparison between two Transformer-based forecasting architectures and discuss the trade-offs between model complexity, forecasting accuracy, and probabilistic performance.

### 7.	Team Responsibilities
- Team Member 1 – Data Engineering and Preprocessing
Responsible for downloading and cleaning the dataset, resampling the data to hourly intervals, handling missing values, normalizing electricity demand, engineering time-based features, generating input-output forecasting sequences, and preparing training, validation, and test datasets.
- Team Member 2 – Baseline Model Development
Responsible for implementing and evaluating baseline forecasting models, including a naive persistence forecast and an LSTM-based forecasting model. This team member will also help establish benchmark performance for comparison with transformer-based models.
- Team Member 3 – Main Transformer Model Development
Responsible for implementing the primary Transformer-based probabilistic forecasting model, including input embeddings, positional encodings, self-attention layers, and the forecasting head. This team member will train and tune the main model for multi-step electricity demand prediction.
- Team Member 4 – Probabilistic Forecasting and Evaluation
Responsible for implementing the probabilistic forecasting framework, including quantile regression, pinball loss, prediction interval generation, and uncertainty evaluation metrics. This team member will also contribute to comparative analysis of model outputs and interpretation of probabilistic forecasting results.

If time permits, the optional implementation of a second Transformer-based architecture will be carried out collaboratively, with the work divided equally among all four team members.

