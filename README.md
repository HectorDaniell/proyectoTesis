# Computer Vision-Based System for Gross Motor Skills Assessment

[![Python](https://img.shields.io/badge/Python-3.9.6-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.5-orange.svg)](https://mediapipe.dev/)

## 🎯 Description

This project implements an intelligent system that uses computer vision and machine learning to assess gross motor skills performance in exercises performed by children with Down syndrome. The system analyzes video recordings of physical exercises and provides quantitative performance assessments using advanced pose detection and machine learning techniques.

### Key Technologies
- **Computer Vision**: MediaPipe for real-time pose detection
- **Machine Learning**: Random Forest, SVM, and other classifiers
- **Data Processing**: PCA for dimensionality reduction
- **Video Processing**: OpenCV for video analysis
- **Data Analysis**: Pandas and NumPy for data manipulation

## ✨ Features

- **Real-time Pose Detection**: Accurate body landmark extraction using MediaPipe
- **Multi-Exercise Support**: Jump, crawl, sit, and ball throwing exercises
- **Performance Classification**: Three-level performance assessment (High, Moderate, Low)
- **Automated Training Pipeline**: End-to-end model training and evaluation
- **Comprehensive Testing**: Functional and non-functional test suites
- **Modular Architecture**: Extensible design for new exercise types

## 🏗️ Architecture

The system follows a modular architecture with the following components:

```
proyectoTesis/
├── src/
│   ├── training/          # Model training pipeline
│   │   ├── main_training.py
│   │   ├── process_videos.py
│   │   ├── pca_reduction.py
│   │   ├── train_model.py
│   │   ├── compare_models.py
│   │   └── label_data_*.py
│   └── evaluation/        # Model evaluation pipeline
│       ├── main_evaluation.py
│       └── predict_performance.py
├── data/
│   ├── raw/              # Input video files organized by exercise
│   │   ├── crawl/        # Crawl exercise videos
│   │   ├── jump/         # Jump exercise videos
│   │   ├── sit/          # Sit exercise videos
│   │   └── throw/        # Throw exercise videos
│   ├── processed/        # Processed landmark data
│   ├── models/           # Trained model files (.pkl)
│   └── results/          # Evaluation and comparison results
│       └── confusion_matrices/  # Confusion matrix visualizations
```

## 📋 Prerequisites

- **Python**: 3.9.6 or higher
- **Git**: For version control (optional)
- **Camera**: Compatible camera for testing (optional)
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: At least 2GB free space

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HectorDaniell/proyectoTesis.git
cd proyectoTesis
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install core dependencies
pip install mediapipe==0.10.5
pip install opencv-contrib-python
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install joblib
pip install numpy
```

## 💻 Usage

### Model Training

To train the model with your video dataset:

```bash
# Navigate to training directory
cd src/training

# Run training pipeline
python main_training.py
```

The training pipeline will:
1. Process video files and extract pose landmarks
2. Label performance based on exercise-specific criteria
3. Apply PCA dimensionality reduction
4. Train and evaluate machine learning models
5. Save trained models to `data/models/`

### Video Evaluation

To evaluate new videos with the trained model:

```bash
# Navigate to evaluation directory
cd src/evaluation

# Run evaluation
python main_evaluation.py
```

The evaluation will:
1. Process the input video
2. Extract pose landmarks
3. Apply the same preprocessing as training
4. Generate performance predictions
5. Display frame-by-frame and overall results

### Running Tests

```bash
# Run comprehensive test suite
python test/test_exercise_modules.py
```

## 📁 Project Structure

```
proyectoTesis/
├── README.md                 # This file
├── src/
│   ├── training/            # Training pipeline modules
│   │   ├── main_training.py        # Main training orchestrator
│   │   ├── process_videos.py       # Video processing and landmark extraction
│   │   ├── pca_reduction.py        # Dimensionality reduction
│   │   ├── train_model.py          # Model training and evaluation
│   │   ├── compare_models.py       # Model comparison utilities
│   │   └── label_data_*.py         # Exercise-specific labeling functions
│   └── evaluation/          # Evaluation pipeline modules
│       ├── main_evaluation.py      # Main evaluation orchestrator
│       └── predict_performance.py  # Performance prediction
├── data/
│   ├── raw/                # Input video files organized by exercise
│   │   ├── crawl/          # Crawl exercise videos (38 videos)
│   │   ├── jump/           # Jump exercise videos (16 videos)
│   │   ├── sit/            # Sit exercise videos (10 videos)
│   │   └── throw/          # Throw exercise videos (35 videos)
│   ├── processed/          # Processed CSV files with landmarks
│   │   ├── *_labeled.csv   # Labeled landmark data
│   │   ├── *_landmarks.csv # Raw landmark data
│   │   └── *_reduced.csv   # PCA-reduced data
│   ├── models/             # Trained model files (.pkl)
│   │   ├── crawl_model.pkl
│   │   ├── jump_model.pkl
│   │   ├── sit_model.pkl
│   │   └── throw_model.pkl
│   └── results/            # Evaluation and comparison results
│       ├── confusion_matrices/    # Confusion matrix visualizations
│       └── model_comparison.csv   # Model performance comparison
└── landmarks_output.csv    # Combined landmark output
```

## 📚 API Documentation

### Training Pipeline

#### `main_training.py`
Main orchestrator for the training pipeline.

```python
def main_training(exercise_name):
    """
    Complete training pipeline for exercise performance classification.
    
    Args:
        exercise_name (str): Name of the exercise to train for
    """
```

#### `process_videos.py`
Handles video processing and landmark extraction.

```python
def process_video(video_path, exercise_name, combined_df):
    """
    Extract pose landmarks from video using MediaPipe.
    
    Args:
        video_path (str): Path to input video
        exercise_name (str): Exercise type
        combined_df (pd.DataFrame): Existing landmark data
        
    Returns:
        pd.DataFrame: Combined landmark data
    """
```

### Evaluation Pipeline

#### `predict_performance.py`
Main prediction and evaluation functions.

```python
def predict_performance(video_path, model_path, pca_components):
    """
    Complete pipeline for predicting exercise performance.
    
    Args:
        video_path (str): Path to input video
        model_path (str): Path to trained model
        pca_components (int): Number of PCA components
    """
```

## 🧪 Testing

The project includes comprehensive testing covering:

- **Functional Tests**: Model validation, input/output processing
- **Performance Tests**: Processing speed, memory usage, FPS analysis
- **Robustness Tests**: Error handling, edge cases

Run the test suite:

```bash
python test/test_exercise_modules.py
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include tests for new features
- Update documentation as needed

## 📊 Performance Metrics

The system achieves the following performance metrics:

- **Accuracy**: >75% on test datasets
- **Processing Speed**: 15+ FPS on standard hardware
- **Memory Usage**: Optimized for 4GB+ systems
- **Supported Exercises**: Jump, Crawl, Sit, Ball Throwing

## 🔧 Troubleshooting

### Common Issues

**MediaPipe Installation Issues:**
```bash
# Try installing specific version
pip install mediapipe==0.10.5
```

**Memory Issues:**
- Reduce video resolution
- Process videos in smaller batches
- Increase system RAM

**Performance Issues:**
- Use GPU acceleration if available
- Reduce PCA components
- Optimize video frame rate

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

For questions, improvements, or collaboration:

- **Email**: [hector.oviedo1312@gmail.com](mailto:hector.oviedo1312@gmail.com)
- **GitHub**: [https://github.com/HectorDaniell](https://github.com/HectorDaniell)

---

**Note**: This project is part of a research thesis on computer vision applications in motor skills assessment for children with special needs.
