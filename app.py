import datetime
import gzip
import io
import os
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class DynamicNN(nn.Module):

    def __init__(self, input_dim, hidden_layers, activation_function, dropout_rate=0.5):
        """Initializes the dynamic neural network."""
        super(DynamicNN, self).__init__()
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.model = self._initialize_model(input_dim, hidden_layers)

    def _initialize_model(self, input_dim, hidden_layers, num_neurons=128):
        """Initializes the layers of the neural network."""
        activation_function = self._get_activation_function()
        layers = []

        layers.append(nn.Linear(input_dim, num_neurons))
        layers.append(nn.BatchNorm1d(num_neurons))
        layers.append(activation_function)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.BatchNorm1d(num_neurons))
            layers.append(activation_function)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(num_neurons, 1))

        model = nn.Sequential(*layers)
        self._init_weights(model)
        return model

    def _get_activation_function(self):
        """Returns the specified activation function module."""
        if self.activation_function == "Swish":
            return nn.SiLU()
        elif self.activation_function == "GELU":
            return nn.GELU()
        elif self.activation_function == "Leaky ReLU":
            return nn.LeakyReLU()
        elif self.activation_function == "ELU":
            return nn.ELU()
        elif self.activation_function == "SELU":
            return nn.SELU()
        else:
            raise ValueError("Invalid activation function selected")

    def _init_weights(self, model):
        """Initializes weights and biases for linear layers."""
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Performs a forward pass through the network."""
        return self.model(x)


def select_optimizer(model):
    """Selects and configures the optimizer based on user choice."""
    optimizer_name = st.selectbox(
        "🛠️ Select optimizer", ["Adam", "Adagrad", "RMSprop", "AdamW", "Adamax"]
    )

    lr = st.number_input("Learning Rate", value=0.001, format="%f")

    optimizer = None
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        alpha = st.number_input("RMSprop Alpha", value=0.99, format="%f")
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)
    elif optimizer_name == "AdamW":
        weight_decay = st.number_input("AdamW Weight Decay", value=1e-2, format="%e")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=lr)

    return optimizer


def remove_old_models():
    """Removes old model files from the 'models' directory."""
    if os.path.exists("models"):
        for model_file in os.listdir("models"):
            file_path = os.path.join("models", model_file)
            if model_file.endswith((".pkl", ".zip", ".gz", ".pt")) and os.path.isfile(
                file_path
            ):
                os.remove(file_path)


def train_model(model, train_loader, criterion, optimizer, epochs=100, device="cpu"):
    """Trains the neural network model."""
    model.train()
    losses = []
    progress_bar = st.progress(0, text="Training progress: 0%")
    epoch_text = st.empty()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    remove_old_models()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        scheduler.step()

        progress_bar.progress(
            (epoch + 1) / epochs,
            text=f"Training progress: {int((epoch + 1) / epochs * 100)}%",
        )
        epoch_text.text(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    return losses


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluates the trained neural network model."""
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs.float())
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())
    return actuals, predictions


def plot_results(
    actuals,
    predictions,
    title="Actual vs Predicted Values",
    xaxis_title="Sample",
    yaxis_title="Value",
):
    """Plots actual vs. predicted values with customizable plot parameters."""
    trace1 = go.Scatter(y=actuals[:150], mode="lines", name="Actual", fill="tozeroy")
    trace2 = go.Scatter(
        y=predictions[:150], mode="lines", name="Predicted", fill="tozeroy"
    )
    layout = go.Layout(
        title=title, xaxis=dict(title=xaxis_title), yaxis=dict(title=yaxis_title)
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)


def plot_losses(losses):
    """Plots the training loss over epochs."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode="lines", name="Loss"))
    fig.update_layout(
        title="Training Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss"
    )
    st.plotly_chart(fig)


def calculate_metrics(actuals, predictions):
    """Calculates RMSE, MAE, and R-squared metrics."""
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    rmse = np.linalg.norm(actuals - predictions) / np.sqrt(len(actuals))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    return rmse, mae, r2


def handle_errors(func):
    """Decorator to catch and display exceptions in Streamlit."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error("An error occurred. Please try again later.")
            st.write(f"Error details: {str(e)}")

    return wrapper


st.set_page_config(page_title="NUMLP", layout="wide")
st.title("NUMLP")
st.header("Numeric Prediction with Neural Networks")
with st.expander("Project Description"):
    st.markdown(
        """
numlp is a Neural Network fueled predictor. Uses uploaded excel, labeled header + numeric content to predict selected output value.
"""
    )

if "model_info" not in st.session_state:
    st.session_state["model_info"] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_original = None

with st.sidebar:
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        try:
            df_original = pd.read_excel(uploaded_file)
            st.sidebar.markdown("---")
            st.write("Data Summary:")
            st.write(f"**Row number:** {df_original.shape[0]}")
            st.write(f"**Column number:** {df_original.shape[1]}")
            st.write(f"**Column Names:** {', '.join(df_original.columns)}")
            st.write("**First rows:**")
            st.dataframe(df_original.head())
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            df_original = None

    st.sidebar.markdown("---")

    st.header("Model Information")
    if "model_info" in st.session_state:
        info = st.session_state["model_info"]
        if info:
            st.write(f"**Hidden Layers:** {info.get('hidden_layers', 'N/A')}")
            st.write(f"**Epochs:** {info.get('epochs', 'N/A')}")
            st.write(f"**Learning Rate:** {info.get('learning_rate', 'N/A')}")
            st.write(f"**Batch Size:** {info.get('batch_size', 'N/A')}")
            st.write(f"**Total Parameters:** {info.get('total_parameters', 'N/A')}")
            st.write(
                f"**Activation Function:** {info.get('activation_function', 'N/A')}"
            )
            st.write(f"**Dropout Rate:** {info.get('dropout_rate', 'N/A')}")
            st.write(f"**Optimizer:** {info.get('optimizer_name', 'N/A')}")
            st.write(f"**Device:** {device.type.upper()}")
        else:
            st.write("No model information available.")


def enable_dropout_on_model(model):
    """Enables dropout layers during inference for Monte Carlo Dropout."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@handle_errors
def main():
    global df_original

    if uploaded_file is not None and df_original is not None:
        st.write("Headers: ", df_original.columns.tolist())

        numeric_cols = df_original.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.warning(
                "No numeric columns found in the uploaded file. Please upload a file with numeric data for prediction."
            )
            return

        st.info(f"Using only numeric columns for modeling: {', '.join(numeric_cols)}")

        output_header = st.selectbox("📦 Which column is the output?", numeric_cols)

        input_headers = [col for col in numeric_cols if col != output_header]

        if not input_headers:
            st.warning(
                "Not enough numeric columns for training (need at least one input and one output)."
            )
            return

        X = df_original[input_headers].values
        y = df_original[output_header].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train).float(), torch.tensor(y_train).float()
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test).float(), torch.tensor(y_test).float()
        )

        batch_size = st.slider(
            "📊 Select Batch Size", min_value=16, max_value=128, value=32, step=16
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with st.expander("🛠️ Advanced options"):
            hidden_layers = st.slider(
                "📚 Select number of hidden layers",
                min_value=3,
                max_value=10,
                value=3,
                step=1,
            )
            num_neurons = st.slider(
                "🧠 Select number of neurons per hidden layer",
                min_value=32,
                max_value=512,
                value=128,
                step=32,
            )
            epochs = st.slider(
                "🏋️‍♂️ Select number of epochs",
                min_value=100,
                max_value=1000,
                value=100,
                step=1,
            )

            dropout_rate = st.slider(
                "🚫 Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05
            )

            activation_function_links = {
                "Leaky ReLU": "https://en.wikipedia.org/wiki/Rectifier_(neural_networks)",
                "Swish": "https://en.wikipedia.org/wiki/Swish_function",
                "GELU": "https://en.wikipedia.org/wiki/Activation_function",
                "ELU": "https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELU",
                "SELU": "https://en.wikipedia.org/wiki/Activation_function",
            }
            activation_function = st.selectbox(
                "🔧 Select activation function", list(activation_function_links.keys())
            )
            st.write(
                f"Read about {activation_function}: [Here]({activation_function_links[activation_function]})"
            )

            model = DynamicNN(
                input_dim=len(input_headers),
                hidden_layers=hidden_layers,
                activation_function=activation_function,
                dropout_rate=dropout_rate,
            ).to(device)
            optimizer = select_optimizer(model)

        st.session_state["model_info"] = {
            "hidden_layers": hidden_layers,
            "epochs": epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": batch_size,
            "activation_function": activation_function,
            "dropout_rate": dropout_rate,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "epochs_trained": 0,
            "optimizer_name": type(optimizer).__name__,
            "device": device.type.upper(),
        }
        st.session_state["df_original"] = df_original
        st.session_state["output_header"] = output_header
        st.session_state["input_headers"] = input_headers

        criterion = nn.MSELoss()

        if st.button("Train the model"):
            st.subheader("🚀 Model Training Initiated")
            st.info(
                f"Training the model for {epochs} epochs with {hidden_layers} hidden layers using {activation_function} activation function on {device.type.upper()}..."
            )

            with st.spinner("Training in progress... This might take a while."):
                losses = train_model(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    epochs=epochs,
                    device=device,
                )

            st.subheader("📈 Training Performance")
            plot_losses(losses)

            st.subheader("📊 Model Evaluation")
            actuals, predictions = evaluate_model(model, test_loader, device=device)
            plot_results(actuals, predictions)

            rmse, mae, r2 = calculate_metrics(actuals, predictions)
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
            st.metric(label="R-squared (R²)", value=f"{r2:.2f}")

            st.success("🎉 Training completed successfully!")

            os.makedirs("models", exist_ok=True)
            prefix = output_header[:10].upper()
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
            model_filename = (
                f"models/{prefix}_{current_time}_{str(uuid.uuid4())[:6]}.pt"
            )

            torch.save(model.state_dict(), model_filename)

            st.session_state["model_trained"] = True
            st.session_state["model_filename"] = model_filename
            st.session_state["model"] = model
            st.session_state["model_info"]["epochs_trained"] = epochs

    if "model_trained" in st.session_state and st.session_state["model_trained"]:
        model_filename = st.session_state["model_filename"]

        if os.path.exists(model_filename):
            compressed_model = io.BytesIO()
            with gzip.GzipFile(fileobj=compressed_model, mode="wb") as gf:
                with open(model_filename, "rb") as f:
                    gf.write(f.read())

            compressed_model.seek(0)

            st.download_button(
                label="Download model (compressed)",
                data=compressed_model,
                file_name=os.path.basename(model_filename),
                mime="application/gzip",
            )
        else:
            st.error(f"Model file not found: {model_filename}. Cannot download.")

    if "model" in st.session_state:
        model = st.session_state["model"]
        input_headers = st.session_state["input_headers"]
        output_header = st.session_state["output_header"]
        df_original = st.session_state["df_original"]
        current_device_info = st.session_state["model_info"].get("device", "CPU")

        st.subheader(f"🔮 Make a Prediction (using {current_device_info})")

        if "input_values" not in st.session_state:
            st.session_state["input_values"] = {header: 0.0 for header in input_headers}

        input_values = []
        for header in input_headers:
            value = st.number_input(
                f"Value for **{header}**",
                key=f"predict_input_{header}",
                value=st.session_state["input_values"][header],
            )
            st.session_state["input_values"][header] = value
            input_values.append(value)

        if st.button("Predict Single Value"):
            model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor([input_values]).float().to(device)
                prediction = model(input_tensor).item()
            st.info(
                f"Predicted output value for **{output_header}**: **{prediction:.4f}**"
            )

            last_historical_x = df_original.index.max()
            last_historical_y = df_original[output_header].iloc[-1]

            fig_single_pred = go.Figure()
            fig_single_pred.add_trace(
                go.Scatter(
                    x=df_original.index[-50:],
                    y=df_original[output_header].iloc[-50:],
                    mode="lines+markers",
                    name=f"Historical {output_header}",
                    line=dict(color="blue"),
                )
            )
            fig_single_pred.add_trace(
                go.Scatter(
                    x=[last_historical_x, last_historical_x + 1],
                    y=[last_historical_y, prediction],
                    mode="lines+markers",
                    name="Single Prediction",
                    line=dict(color="red", dash="dot"),
                    marker=dict(size=8, symbol="star"),
                )
            )
            fig_single_pred.update_layout(
                title=f"Historical {output_header} and Single Prediction",
                xaxis_title="Index/Time",
                yaxis_title=output_header,
            )
            st.plotly_chart(fig_single_pred)

        st.markdown("---")
        st.subheader("🗓️ Predict a Range of Future Values")

        last_actual_input_row = df_original[input_headers].iloc[-1].values.tolist()

        all_output_values = df_original[output_header].tolist()
        last_actual_output_idx = len(all_output_values) - 1

        days_to_predict = st.slider(
            "Number of future steps to predict", min_value=1, max_value=30, value=7
        )

        num_monte_carlo_passes = st.slider(
            "Number of Monte Carlo Passes for Uncertainty",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )

        if st.button("Predict Range"):
            st.info(
                f"Predicting {days_to_predict} future values for {output_header} with {num_monte_carlo_passes} Monte Carlo passes..."
            )

            # Enable dropout layers for MC Dropout
            enable_dropout_on_model(model)

            all_future_predictions_mc = []

            for _ in range(num_monte_carlo_passes):
                single_mc_run_predictions = []
                current_input_for_recursive_pred = (
                    torch.tensor([last_actual_input_row]).float().to(device)
                )

                with torch.no_grad():
                    for _ in range(days_to_predict):
                        predicted_value = model(current_input_for_recursive_pred).item()
                        single_mc_run_predictions.append(predicted_value)
                all_future_predictions_mc.append(single_mc_run_predictions)

            # Convert list of lists to numpy array for easier calculation
            all_future_predictions_mc = np.array(all_future_predictions_mc)

            # Calculate mean and standard deviation for each future step
            mean_predictions = np.mean(all_future_predictions_mc, axis=0)
            std_predictions = np.std(all_future_predictions_mc, axis=0)

            # Define confidence interval (e.g., 95% confidence interval using 1.96 * std)
            lower_bound = mean_predictions - 1.96 * std_predictions
            upper_bound = mean_predictions + 1.96 * std_predictions

            future_x_indices = np.arange(
                last_actual_output_idx + 1, last_actual_output_idx + 1 + days_to_predict
            )

            fig_range_pred = go.Figure()
            fig_range_pred.add_trace(
                go.Scatter(
                    x=list(range(len(all_output_values))),
                    y=all_output_values,
                    mode="lines",
                    name=f"Historical {output_header}",
                    line=dict(color="blue"),
                )
            )
            fig_range_pred.add_trace(
                go.Scatter(
                    x=future_x_indices,
                    y=mean_predictions,
                    mode="lines+markers",
                    name="Future Mean Prediction",
                    line=dict(color="red", dash="dash"),
                    marker=dict(size=6, symbol="circle"),
                )
            )
            fig_range_pred.add_trace(
                go.Scatter(
                    x=np.concatenate([future_x_indices, future_x_indices[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.1)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Confidence Interval",
                    showlegend=True,
                )
            )

            fig_range_pred.add_vline(
                x=last_actual_output_idx,
                line_width=1,
                line_dash="dash",
                line_color="green",
                annotation_text="Start of Predictions",
                annotation_position="top right",
            )

            fig_range_pred.update_layout(
                title=f"Historical {output_header} and Future Predictions with Uncertainty",
                xaxis_title="Index/Time",
                yaxis_title=output_header,
                hovermode="x unified",
            )
            st.plotly_chart(fig_range_pred)

            # Revert model to eval mode for consistency after MC Dropout passes
            model.eval()


st.sidebar.markdown(
    """
    ---
    Created by [th3pajay](https://github.com/th3pajay) 
    ![UserGIF](https://user-images.githubusercontent.com/74038190/219925470-37670a3b-c3e2-4af7-b468-673c6dd99d16.png)
"""
)

main()
