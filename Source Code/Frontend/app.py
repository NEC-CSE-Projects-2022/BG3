import os
from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from scipy.stats import entropy  # for uncertainty check

# ==============================
# CONFIG
# ==============================
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = (128, 128)
users = {}  # Mock DB

CONF_THRESHOLD = 0.7
UNCERTAINTY_THRESHOLD = 0.4

# ====== Load Trained Model ======
MODEL_PATH = "DenseNet121_LSTM_GRU_final.h5"
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"⚠️ Could not load model: {e}")

# ====== Class Names ======
CLASS_NAMES = ["Parasitized", "Uninfected"]

# ====== DenseNet121 Feature Extractor ======
# build feature extractor for 128x128 input
try:
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False
    out = GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=out)
    print("✅ DenseNet121 feature extractor ready (input 128x128).")
except Exception as e:
    feature_extractor = None
    print(f"⚠️ Could not initialize DenseNet121 feature extractor: {e}")

# ==============================
# PREPROCESSING FUNCTIONS
# ==============================
def resize_image(img, size=IMG_SIZE):
    return cv2.resize(img, size)

def denoise_image(img):
    # Non-local means denoising for colored images
    try:
        return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        # fallback: return original if method fails
        return img

def apply_clahe(img):
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception:
        return img

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    try:
        return cv2.filter2D(img, -1, kernel)
    except Exception:
        return img

def normalize_image(img):
    # returns float32 image scaled 0..1
    return img.astype("float32") / 255.0

def preprocess_pipeline(image_path):
    """
    Runs preprocessing pipeline on uploaded image.
    Returns:
      - display_img_uint8: processed RGB image in uint8 (0-255) for saving/preview
      - model_input_array: normalized float32 image array shaped (1, H, W, 3) for model
    """
    # Load with PIL backend via keras utils, then convert to numpy RGB
    pil_img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(pil_img).astype("uint8")  # RGB order

    # OpenCV expects uint8 in RGB if we use cv2 functions with cvtColor accordingly
    img = arr.copy()
    img = resize_image(img)
    img = denoise_image(img)
    img = apply_clahe(img)
    img = sharpen_image(img)

    # Save displayable image as uint8 RGB
    display_img_uint8 = np.clip(img, 0, 255).astype("uint8")

    # Prepare model input (normalized)
    model_input = normalize_image(display_img_uint8)
    model_input_array = np.expand_dims(model_input, axis=0)  # shape (1,H,W,3)

    return display_img_uint8, model_input_array

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("Home.html", title="Home")
@app.route("/dataset")
def dataset():
    return render_template("Dataset.html", title="Dataset")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not username or not email or not password:
            flash("Please fill all fields.", "warning")
            return redirect(request.url)
        if email in users:
            flash("Email already registered.", "warning")
            return redirect(request.url)
        users[email] = {"username": username, "password": password}
        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("Signup.html", title="Signup")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        user = users.get(email)
        if user and user.get("password") == password:
            flash("Login successful.", "success")
            return redirect(url_for("predict"))
        else:
            flash("Invalid email or password", "danger")
            return redirect(request.url)
    return render_template("Login.html", title="Login")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    uploaded_img = None
    processed_img_name = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!", "warning")
            return redirect(request.url)

        # save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        uploaded_img = filename

        # if model or feature_extractor not available, notify user but still save processed preview
        if model is None or feature_extractor is None:
            # create processed preview anyway
            try:
                processed_img_uint8, _ = preprocess_pipeline(file_path)
                processed_img_name = "processed_" + filename
                processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_img_name)
                # OpenCV expects BGR for imwrite; our image is RGB
                cv2.imwrite(processed_path, cv2.cvtColor(processed_img_uint8, cv2.COLOR_RGB2BGR))
            except Exception as e:
                flash(f"Error during preprocessing: {e}", "danger")
            flash("Model or feature extractor not available. Prediction cannot be performed.", "warning")
            return render_template("Predict.html", title="Predict", result=None,
                                   uploaded_img=uploaded_img, processed_img=processed_img_name)

        # Run preprocessing and model inference
        try:
            processed_img_uint8, model_input = preprocess_pipeline(file_path)

            # Extract features via DenseNet
            features = feature_extractor.predict(model_input)
            # Some architectures expect temporal dimension; your original code did expand_dims(features, axis=1).
            # Keep the same shape as model expects. Here we try to match your previous code:
            features_for_model = np.expand_dims(features, axis=1)

            # Predict
            preds = model.predict(features_for_model)
            # preds can be (1, N) or (1, 2). Ensure we get 1D probability vector.
            probs = np.squeeze(preds)
            # If model returns logits, convert to softmax
            if probs.ndim == 0:
                # scalar, make into vector
                probs = np.array([probs])
            if probs.sum() <= 1.0001 and probs.sum() >= 0.9999:
                # seems already probabilities
                pass
            else:
                # apply softmax to get probabilities
                exp = np.exp(probs - np.max(probs))
                probs = exp / exp.sum()

            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
            confidence = float(np.max(probs))
            uncertainty = float(entropy(probs))  # entropy measure

            # Threshold check
            if confidence < CONF_THRESHOLD or uncertainty > UNCERTAINTY_THRESHOLD:
                result = {
                    "prediction": "Not a Malaria Cell",
                    "is_cell": False,
                    "confidence": round(confidence, 3),
                    "uncertainty": round(uncertainty, 3),
                    "description": "The uploaded image does not resemble a recognized malaria cell.",
                    "note": "Try uploading a clear microscopic blood smear image."
                }
            else:
                descriptions = {
                    "Parasitized": "This cell shows presence of malaria parasites.",
                    "Uninfected": "This cell does not show malaria parasites."
                }
                result = {
                    "prediction": pred_class,
                    "is_cell": True,
                    "confidence": round(confidence, 3),
                    "uncertainty": round(uncertainty, 3),
                    "description": descriptions.get(pred_class, "No description available."),
                    "note": "Please consult a medical professional for confirmation."
                }

            # Save processed image for preview
            processed_img_name = "processed_" + filename
            processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_img_name)
            cv2.imwrite(processed_path, cv2.cvtColor(processed_img_uint8, cv2.COLOR_RGB2BGR))

            return render_template("Predict.html",
                                   title="Predict",
                                   result=result,
                                   uploaded_img=uploaded_img,
                                   processed_img=processed_img_name)

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")
            return redirect(request.url)

    # GET request
    return render_template("Predict.html", title="Predict", result=result,
                           uploaded_img=uploaded_img, processed_img=processed_img_name)


# ==============================
# MAIN - RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
