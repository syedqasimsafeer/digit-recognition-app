
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import plotly.express as px

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


@st.cache_resource
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image):
    """
    Convert any uploaded or drawn image into MNIST style:
    grayscale, 28x28, normalized, shape [1, 1, 28, 28].
    """

    image = image.convert("L")

    # Resize first to keep deployment simple
    image = image.resize((28, 28))

    image_array = np.array(image).astype(np.float32)

    # Normalize to 0 to 1
    image_array = image_array / 255.0

    # MNIST style is white digit on black background.
    # If background is white, invert it.
    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array

    # Increase contrast slightly
    image_array = np.clip(image_array * 1.2, 0, 1)

    tensor = torch.tensor(image_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor, image_array


def predict_digit(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1).numpy()[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return predicted_digit, confidence, probabilities


def main():
    st.set_page_config(
        page_title="Digit Recognition App",
        page_icon="🔢",
        layout="wide"
    )

    st.title("🔢 Digit Recognition App")
    st.write(
        "A deep learning project using MNIST, NumPy neural network concepts, "
        "PyTorch CNN training, and Streamlit deployment."
    )

    model = load_model()

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "MNIST")
    col2.metric("Image Size", "28 × 28")
    col3.metric("Classes", "10 digits")

    tab1, tab2, tab3 = st.tabs([
        "🔮 Predict Digit",
        "📚 Project Explanation",
        "ℹ️ About"
    ])

    with tab1:
        st.subheader("Choose input method")

        input_method = st.radio(
            "Input method",
            ["Upload Image", "Draw Digit"],
            horizontal=True
        )

        image_to_predict = None

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a handwritten digit image",
                type=["png", "jpg", "jpeg"]
            )

            if uploaded_file is not None:
                image_to_predict = Image.open(uploaded_file)

        else:
            if CANVAS_AVAILABLE:
                st.write("Draw a digit using white color on black background.")

                canvas_result = st_canvas(
                    fill_color="black",
                    stroke_width=20,
                    stroke_color="white",
                    background_color="black",
                    width=280,
                    height=280,
                    drawing_mode="freedraw",
                    key="canvas"
                )

                if canvas_result.image_data is not None:
                    image_to_predict = Image.fromarray(
                        canvas_result.image_data.astype("uint8")
                    )
            else:
                st.warning(
                    "Drawing canvas is not available. Please use image upload."
                )

        if image_to_predict is not None:
            tensor, processed_array = preprocess_image(image_to_predict)
            predicted_digit, confidence, probabilities = predict_digit(model, tensor)

            left, right = st.columns([1, 2])

            with left:
                st.subheader("Input Image")
                st.image(image_to_predict, width=250)

                st.subheader("Processed 28×28 Image")
                st.image(processed_array, width=180, clamp=True)

            with right:
                st.subheader("Prediction Result")
                st.success(f"Predicted Digit: {predicted_digit}")
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                probability_df = pd.DataFrame({
                    "Digit": list(range(10)),
                    "Probability": probabilities
                })

                fig = px.bar(
                    probability_df,
                    x="Digit",
                    y="Probability",
                    title="Prediction Probabilities for Digits 0 to 9",
                    text=probability_df["Probability"].round(3)
                )

                fig.update_layout(
                    xaxis=dict(dtick=1),
                    yaxis=dict(range=[0, 1])
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "The model works best with centered, clear, high-contrast handwritten digits."
                )

    with tab2:
        st.subheader("What this project covers")

        st.markdown(
            """
            This project includes three learning parts:

            ### 1. Neural Network from Scratch using NumPy
            - Flattening 28×28 images into 784 input values
            - Hidden layer
            - ReLU activation
            - Softmax output
            - Cross-entropy loss
            - Backpropagation
            - Gradient descent

            ### 2. MLP using PyTorch
            - Fully connected neural network
            - Dropout
            - Training loop
            - Test accuracy

            ### 3. CNN using PyTorch
            - Convolution layers
            - ReLU activation
            - Max pooling
            - Fully connected layer
            - Confusion matrix
            - Wrong prediction analysis
            """
        )

        st.subheader("Why CNN is better for image recognition")
        st.write(
            "A simple MLP treats every pixel as a separate number. A CNN learns local "
            "visual patterns such as edges, curves, and strokes. This is why CNNs "
            "usually perform better for image tasks like handwritten digit recognition."
        )

    with tab3:
        st.subheader("Dataset")
        st.write(
            "This app uses a CNN trained on the MNIST handwritten digit dataset. "
            "MNIST contains grayscale images of handwritten digits from 0 to 9."
        )

        st.subheader("Important Note")
        st.warning(
            "This app is for educational purposes. It works best with MNIST-style digits: "
            "single digit, centered, high contrast, and simple background."
        )


if __name__ == "__main__":
    main()
