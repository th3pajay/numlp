import datetime
import os

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


# Define NN
class DynamicNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_function):
        super(DynamicNN, self).__init__()
        self.activation_function = activation_function
        self.model = self._initialize_model(input_dim, hidden_layers)

    def _initialize_model(self, input_dim, hidden_layers, num_neurons=128):
        activation_function = self._get_activation_function()
        layers = [layer for i in range(hidden_layers) for layer in
                  (nn.Linear(input_dim if i == 0 else num_neurons, num_neurons), activation_function)]
        layers.append(nn.Linear(num_neurons, 1))
        model = nn.Sequential(*layers)
        self._init_weights(model)
        return model

    def _get_activation_function(self):
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
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


# Define optimizer, fine grade training
def select_optimizer(model):
    optimizer_name = st.selectbox(
        "🛠️ Select optimizer",
        ["Adam", "Adagrad", "RMSprop", "AdamW", "Adamax"]
    )

    lr = st.number_input("Learning Rate", value=0.001, format="%f")

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


# Remove obsolete model files
def remove_old_models():
    if os.path.exists('models'):
        for model_file in os.listdir("models"):
            os.remove(os.path.join("models", model_file))


# Train NN
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    losses = []
    progress_bar = st.progress(0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    remove_old_models()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        scheduler.step()
        progress_bar.progress((epoch + 1) / epochs)
    return losses


# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.float())
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())
    return actuals, predictions


# Plot results
def plot_results(actuals, predictions):
    trace1 = go.Scatter(y=actuals[:150], mode='lines', name='Actual', fill='tozeroy')
    trace2 = go.Scatter(y=predictions[:150], mode='lines', name='Predicted', fill='tozeroy')
    layout = go.Layout(title='Actual vs Predicted Values', xaxis=dict(title='Sample'), yaxis=dict(title='Value'))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)


# Plot losses
def plot_losses(losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode='lines', name='Loss'))
    fig.update_layout(title='Training Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig)


# Calculate metrics
def calculate_metrics(actuals, predictions):
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    return rmse, mae, r2


# Catch errors
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            st.error("An error occurred. Please try again later.")
            st.write(f"Error details: {str(e)}")

    return wrapper


# Streamlit frontend
st.set_page_config(page_title="NUMLP", layout="wide")
st.title("NUMLP")
st.header("Numeric Multi-Layer Perceptron predictor")

# Sidebar
if 'model_info' not in st.session_state:
    st.session_state['model_info'] = {}

with st.sidebar:
    # Excel representation
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.sidebar.markdown("---")
        st.write("Data Summary:")
        st.write(f"**Row number:** {df.shape[0]}")
        st.write(f"**Column number:** {df.shape[1]}")
        st.write(f"**Column Names:** {', '.join(df.columns)}")
        st.write("**First rows:**")
        st.dataframe(df.head())

    st.sidebar.markdown("---")

    st.header("Model Information")
    if 'model_info' in st.session_state:
        info = st.session_state['model_info']
        if info:
            st.write(f"**Hidden Layers:** {info.get('hidden_layers', 'N/A')}")
            st.write(f"**Epochs:** {info.get('epochs', 'N/A')}")
            st.write(f"**Learning Rate:** {info.get('learning_rate', 'N/A')}")
            st.write(f"**Batch Size:** {info.get('batch_size', 'N/A')}")
            st.write(f"**Total Parameters:** {info.get('total_parameters', 'N/A')}")
            st.write(f"**Activation Function:** {info.get('activation_function', 'N/A')}")
            st.write(f"**Optimizer:** {info.get('optimizer_name', 'N/A')}")
        else:
            st.write("No model information available.")


@handle_errors
def main():
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Headers: ", df.columns.tolist())

        output_header = st.selectbox("📦 Which column is the output?", df.columns.tolist())
        input_headers = [col for col in df.columns if col != output_header]

        X = df[input_headers].values
        y = df[output_header].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
        test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Advanced options
        with st.expander("🛠️ Advanced options"):
            hidden_layers = st.slider("📚 Select number of hidden layers", min_value=3, max_value=10, value=3, step=1)
            num_neurons = st.slider("🧠 Select number of neurons per hidden layer", min_value=32, max_value=512,
                                    value=128, step=32)
            epochs = st.slider("🏋️‍♂️ Select number of epochs", min_value=100, max_value=1000, value=100, step=1)

            activation_function_links = {
                "Leaky ReLU": "https://en.wikipedia.org/wiki/Rectifier_(neural_networks)",
                "Swish": "https://en.wikipedia.org/wiki/Swish_function",
                "GELU": "https://en.wikipedia.org/wiki/Activation_function",
                "ELU": "https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELU",
                "SELU": "https://en.wikipedia.org/wiki/Activation_function"
            }
            activation_function = st.selectbox("🔧 Select activation function", list(activation_function_links.keys()))
            st.write(f"Read about {activation_function}: [Here]({activation_function_links[activation_function]})")

            # Fine tune model
            model = DynamicNN(input_dim=len(input_headers), hidden_layers=hidden_layers,
                              activation_function=activation_function)
            optimizer = select_optimizer(model)

        # Update session state
        st.session_state['model_info'] = {
            'hidden_layers': hidden_layers,
            'epochs': epochs,
            'learning_rate': 0.001,
            'batch_size': 32,
            'activation_function': activation_function,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'epochs_trained': 0,
            'optimizer_name': optimizer
        }

        criterion = nn.MSELoss()

        if st.button("Train the model"):
            st.write(
                f"Training the model for {epochs} epochs with {hidden_layers} hidden layers using {activation_function} activation function...")
            losses = train_model(model, train_loader, criterion, optimizer, epochs=epochs)

            st.write("Plotting the training losses...")
            plot_losses(losses)

            st.write("Evaluating the model...")
            actuals, predictions = evaluate_model(model, test_loader)

            st.write("Plotting the results...")
            plot_results(actuals, predictions)

            # Metrics
            rmse, mae, r2 = calculate_metrics(actuals, predictions)
            st.write(
                f"Root Mean Squared Error ([RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)): {rmse:.2f}")
            st.write(f"Mean Absolute Error ([MAE](https://en.wikipedia.org/wiki/Mean_absolute_error)): {mae:.2f}")
            st.write(f"R-squared ([R²](https://en.wikipedia.org/wiki/Coefficient_of_determination)): {r2:.2f}")

            # Success
            st.success("🎉 Training completed successfully!")

            # Save model
            prefix = output_header[:10].upper()
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
            model_filename = f"models/{prefix}_{current_time}.pt"

            torch.save(model.state_dict(), model_filename)

            # Update sidebar
            st.session_state['model_trained'] = True
            st.session_state['model_filename'] = model_filename
            st.session_state['model'] = model
            st.session_state['input_headers'] = input_headers
            st.session_state['model_info']['epochs_trained'] = epochs

    # Download model
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        model_filename = st.session_state['model_filename']
        with open(model_filename, 'rb') as f:
            st.download_button(
                label="Download trained model",
                data=f,
                file_name=model_filename,
                mime="application/octet-stream"
            )

    # Predict value
    if 'model' in st.session_state:
        model = st.session_state['model']
        input_headers = st.session_state['input_headers']

        st.write("Input values for prediction:")

        # Initialize input_values in session_state
        if 'input_values' not in st.session_state:
            st.session_state['input_values'] = {header: 0.0 for header in input_headers}

        input_values = []
        for header in input_headers:
            # Use session_state
            value = st.number_input(f"Value for {header}", key=header, value=st.session_state['input_values'][header])
            st.session_state['input_values'][header] = value
            input_values.append(value)

        # Predict based on input_headers
        if st.button("Predict"):
            model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor([input_values]).float()
                prediction = model(input_tensor).item()
            st.write(f"Predicted value for {output_header}: {prediction}")


# Footer
st.sidebar.markdown("""
    ---
    Created by [th3pajay](https://github.com/th3pajay) 
    ![UserGIF](https://user-images.githubusercontent.com/74038190/219925470-37670a3b-c3e2-4af7-b468-673c6dd99d16.png)
""")

main()
