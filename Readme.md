# House Price Prediction Project

## Description
This project implements a machine learning model to predict house prices based on various features such as longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, and ocean proximity. The application provides a user-friendly web interface where users can input these features and get an instant price prediction.

## Features
*   **Web Interface:** A clean and interactive web form for inputting house features.
*   **Two-Column Layout:** Input fields are arranged in a responsive two-column layout for better usability.
*   **Input Validation:** Ensures all required fields are filled before prediction.
*   **Machine Learning Model:** Utilizes a pre-trained model to predict house prices.
*   **Dynamic Prediction Display:** Displays the predicted house price clearly on the webpage.

## Technologies Used
*   **Python:** The primary programming language.
*   **Flask:** Web framework for building the application.
*   **Scikit-learn:** For machine learning model training and prediction.
*   **HTML/CSS:** For the front-end web interface.
*   **Pandas:** For data manipulation.
*   **Numpy:** For numerical operations.

## Installation
To get a copy of this project up and running on your local machine for development and testing purposes, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/House-Price-Project.git
    cd House-Price-Project
    ```
    (Remember to replace `YOUR_USERNAME` with your actual GitHub username and `House-Price-Project` with your repository name.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Run the Flask application:**
    ```bash
    python app.py
    ```
2.  **Open your browser:** Navigate to `http://127.0.0.1:5000/` to access the application.
3.  **Enter house features:** Fill in the required fields in the form.
4.  **Get Prediction:** Click the "predict" button to see the estimated house price.

## File Structure
*   `app.py`: The main Flask application, handling routes and predictions.
*   `Model_Pipeline.py`: Contains ML model and a data preprocessing pipeline.
*   `housing.csv`: The dataset used for training and testing.
*   `Model.pkl`: The trained machine learning model (e.g., a regression model).
*   `Pipeline.pkl`: The data preprocessing pipeline (e.g., scalers, encoders).
*   `requirements.txt`: Lists the Python dependencies required for the project.
*   `static/`:
    *   `style.css`: Custom CSS for styling the web application.
*   `templates/`:
    *   `index.html`: The main HTML template for the web interface.

## Obtaining Model and Pipeline Files

The `Model.pkl` and `Pipeline.pkl` files are essential for the application to function, but they are not included in the repository due to size constraints. You can obtain these files using one of the following methods:

1.  **Run the `Model_Pipeline.py` script:**
    Execute the `Model_Pipeline.py` script to train the model and generate both `Model.pkl` and `Pipeline.pkl` files in your local directory.
    ```bash
    python Model_Pipeline.py
    ```

2.  **Download from Google Drive:**
    You can download a zip file containing both `Model.pkl` and `Pipeline.pkl` from the following link: [Google Drive Link](https://drive.google.com/file/d/1dMNyIChV-hQbjchP3Sa0Z5xnO_gaoPZS/view?usp=sharing)

## Prediction Model Details
The prediction model used in this project is a **Scikit-learn based regression model** (likely trained using a dataset like `housing.csv`). It was trained on the `housing.csv` dataset. The data preprocessing steps, including **feature scaling and one-hot encoding for categorical features**, are handled by `Pipeline.pkl`.

## License
This project is licensed under the **MIT License** - see the LICENSE.md file for details.