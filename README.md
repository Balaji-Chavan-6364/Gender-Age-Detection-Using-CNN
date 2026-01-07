# Gender and Age Detection System

A Deep Learning project that accurately identifies the gender and age range of a person from a single image or video stream. This project offers two interfaces: a **Flask Web Application** for easy browser access and history tracking, and a **Command Line / Desktop Script** for real-time local detection.

## 🚀 Features

*   **Dual Interface**:
    *   **Web App**: User-friendly interface to upload images or capture from camera, view results, and store detection history in a database.
    *   **Desktop Script**: Lightweight script for running detections on local image files or live webcam feed.
*   **Deep Learning Models**: Utilizes pre-trained Caffe models for robust Gender and Age classification.
*   **Face Detection**: accurate face detection using TensorFlow's DNN model.
*   **Database Integration**: The web app saves detection records (Name, Gender, Age, Timestamp) to a SQLite database.

## 🛠️ Technology Stack

*   **Language**: Python 3.x
*   **Web Framework**: Flask
*   **Computer Vision**: OpenCV (`opencv-python`)
*   **Data Processing**: NumPy
*   **Database**: SQLite
*   **Models**:
    *   Face Detection: TensorFlow (`.pb`)
    *   Age & Gender: Caffe (`.caffemodel`)

## 📋 Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

### Dependencies
The project relies on the following key libraries:
*   Flask
*   opencv-python
*   numpy
*   Werkzeug

## ⚙️ Installation

1.  **Clone or Download** this repository to your local machine.
2.  **Navigate** to the project directory:
    ```bash
    cd Gender-and-Age-Detection-master
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Option 1: Web Application (Recommended)
The web application provides a full UI with history tracking.

1.  **Run the application**:
    ```bash
    python app.py
    ```
2.  **Open your browser** and go to:
    ```
    http://localhost:5000
    ```
3.  **Use the App**:
    *   Enter a name.
    *   Upload an image OR use the camera capture feature.
    *   View the processed image with age and gender labels.
    *   Records are automatically saved to `database.db`.

### Option 2: Command Line / Desktop Script
Useful for quick tests or local real-time webcam inference.

**To detect from an image file:**
```bash
python detect.py --image path/to/your/image.jpg
```
*Example:* `python detect.py --image girl1.jpg`

**To detect from Webcam (Real-time):**
```bash
python detect.py
```
*   Press **Ctrl + C** in the terminal or close the window to stop.

## 📂 Project Structure

```
├── app.py                      # Main Flask application entry point
├── detect.py                   # Standalone script for CLI/Desktop usage
├── database.db                 # SQLite database (created on first run of app.py)
├── requirements.txt            # Python dependencies
├── static/                     # Web assets (CSS, JS, Uploads)
│   ├── uploads/                # Stores original uploaded images
│   └── processed/              # Stores images with detection overlays
├── templates/                  # HTML templates for Flask
├── models files...             # (.pb, .pbtxt, .prototxt, .caffemodel)
└── README.md                   # Project documentation
```

## 🧠 Model Details

This project uses models trained by **Tal Hassner and Gil Levi**.
*   **Gender Classes**: Male, Female
*   **Age Ranges**: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)

*Note: Age prediction is treated as a classification problem (assigning an age bucket) rather than a regression problem due to the difficulty of pinpointing exact age from visual features alone.*

## 📄 License & Credits

*   Models trained by [Tal Hassner and Gil Levi](https://talhassner.github.io/home/projects/Adience/Adience-data.html).
*   Adience Benchmark Dataset used for training.
