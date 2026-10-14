# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import json
# from pathlib import Path


# # --------------------------------------------------
# # Page Configuration
# # --------------------------------------------------

# st.set_page_config(
#     page_title="Classroom Attendance Prediction",
#     page_icon="📊",
#     layout="wide"
# )


# # --------------------------------------------------
# # Project Paths
# # --------------------------------------------------

# BASE_DIR = Path(__file__).resolve().parent.parent

# MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
# FEATURE_PATH = BASE_DIR / "models" / "feature_information.json"


# # --------------------------------------------------
# # Load Model
# # --------------------------------------------------

# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)


# @st.cache_data
# def load_feature_information():
#     with open(FEATURE_PATH, "r") as file:
#         return json.load(file)


# model = load_model()
# feature_information = load_feature_information()


# # --------------------------------------------------
# # Page Header
# # --------------------------------------------------

# st.title("📊 Classroom Attendance Prediction")

# st.write(
#     "Predict the expected classroom attendance percentage "
#     "using academic schedule and historical attendance information."
# )

# st.divider()


# # --------------------------------------------------
# # Input Section
# # --------------------------------------------------

# st.subheader("Enter Lecture Information")

# col1, col2, col3 = st.columns(3)


# with col1:

#     day = st.selectbox(
#         "Day",
#         [
#             "Monday",
#             "Tuesday",
#             "Wednesday",
#             "Thursday",
#             "Friday",
#             "Saturday"
#         ]
#     )

#     lecture_number = st.number_input(
#         "Lecture Number",
#         min_value=1,
#         max_value=20,
#         value=1
#     )

#     semester = st.number_input(
#         "Semester",
#         min_value=1,
#         max_value=6,
#         value=3
#     )

#     branch = st.text_input(
#         "Branch",
#         value="MCA"
#     )

#     section = st.text_input(
#         "Section",
#         value="A"
#     )


# with col2:

#     subject = st.text_input(
#         "Subject",
#         value="Mobile Application Development"
#     )

#     faculty_id = st.text_input(
#         "Faculty ID",
#         value="F_001"
#     )

#     classroom = st.text_input(
#         "Classroom",
#         value="401"
#     )

#     total_students = st.number_input(
#         "Total Enrolled Students",
#         min_value=1,
#         max_value=200,
#         value=30
#     )

#     previous_attendance = st.number_input(
#         "Previous Lecture Attendance (%)",
#         min_value=0.0,
#         max_value=100.0,
#         value=80.0
#     )


# with col3:

#     faculty_experience = st.number_input(
#         "Faculty Experience Years",
#         min_value=0,
#         max_value=50,
#         value=5
#     )

#     start_hour = st.number_input(
#         "Start Hour",
#         min_value=0,
#         max_value=23,
#         value=8
#     )

#     day_of_semester = st.number_input(
#         "Day of Semester",
#         min_value=1,
#         max_value=365,
#         value=1
#     )

#     week_number = st.number_input(
#         "Week Number",
#         min_value=1,
#         max_value=52,
#         value=1
#     )

#     rolling_attendance = st.number_input(
#         "Rolling 3 Lecture Attendance (%)",
#         min_value=0.0,
#         max_value=100.0,
#         value=80.0
#     )


# # --------------------------------------------------
# # Additional Inputs
# # --------------------------------------------------

# st.subheader("Lecture Conditions")

# col4, col5, col6 = st.columns(3)


# with col4:

#     practical_theory = st.selectbox(
#         "Practical/Theory",
#         ["Theory", "Practical"]
#     )

#     test_week = st.selectbox(
#         "Test Week",
#         ["No", "Yes"]
#     )

#     assignment_due = st.selectbox(
#         "Assignment Due",
#         ["No", "Yes"]
#     )


# with col5:

#     holiday = st.selectbox(
#         "Holiday",
#         ["No", "Yes"]
#     )

#     weather = st.selectbox(
#         "Weather",
#         [
#             "Normal",
#             "Rainy",
#             "Hot",
#             "Cold"
#         ]
#     )

#     special_event = st.selectbox(
#         "Special Event",
#         ["No", "Yes"]
#     )


# with col6:

#     lecture_occurred = st.selectbox(
#         "Lecture Occurred",
#         ["Yes", "No"]
#     )


# # --------------------------------------------------
# # Prediction
# # --------------------------------------------------

# st.divider()

# if st.button(
#     "🔮 Predict Attendance",
#     type="primary",
#     use_container_width=True
# ):

#     input_data = pd.DataFrame({
#         "Day": [day],
#         "Lecture Number": [lecture_number],
#         "Subject": [subject],
#         "Faculty ID": [faculty_id],
#         "Semester": [semester],
#         "Branch": [branch],
#         "Section": [section],
#         "Classroom": [classroom],
#         "Total Enrolled Students": [total_students],
#         "Previous Lecture Attendance": [previous_attendance],
#         "Practical/Theory": [practical_theory],
#         "Test Week": [test_week],
#         "Assignment Due": [assignment_due],
#         "Holiday": [holiday],
#         "Weather": [weather],
#         "Special Event": [special_event],
#         "Lecture Occurred": [lecture_occurred],
#         "Faculty Experience Years": [faculty_experience],
#         "Previous Attendance Missing": [0],
#         "Start Hour": [start_hour],
#         "Day of Semester": [day_of_semester],
#         "Week Number": [week_number],
#         "Rolling 3 Lecture Attendance": [rolling_attendance]
#     })


#     # Keep only features expected by the trained model
#     expected_features = (
#         feature_information["categorical_features"]
#         + feature_information["numerical_features"]
#     )

#     input_data = input_data.reindex(
#         columns=expected_features
#     )


#     # Prediction
#     prediction = model.predict(input_data)[0]

#     prediction = np.clip(
#         prediction,
#         0,
#         100
#     )


#     # Display result
#     st.success("Prediction completed successfully!")

#     st.metric(
#         label="Predicted Attendance",
#         value=f"{prediction:.2f}%"
#     )


#     # Interpretation
#     if prediction >= 75:

#         st.success(
#             "Expected attendance is good."
#         )

#     elif prediction >= 60:

#         st.warning(
#             "Expected attendance is moderate."
#         )

#     else:

#         st.error(
#             "Expected attendance is low."
#         )

# ============================================================
# CLASSROOM ATTENDANCE PREDICTION SYSTEM
# ============================================================

import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Classroom Attendance Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #ffffff;
        color: #172033;
    }

    .main {
        background-color: #ffffff;
    }

    h1, h2, h3 {
        color: #172033 !important;
    }

    p, label, span {
        color: #40516d;
    }

    .subtitle {
        font-size: 18px;
        color: #52627a;
        margin-bottom: 25px;
    }

    .divider {
        height: 1px;
        background-color: #e2e6eb;
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .prediction-card {
        background-color: #f5f8ff;
        border: 1px solid #cddcff;
        border-radius: 15px;
        padding: 25px;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 700;
        color: #1f5eff;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 48px;
        font-size: 17px;
        font-weight: 600;
        background-color: #ff4b4b;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background-color: #e63939;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dfe3e8;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(APP_DIR)

ENGINEERED_DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "cleaned",
    "attendance_engineered.csv"
)

CLEANED_DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "cleaned",
    "attendance_cleaned.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_model.pkl"
)

MODEL_COMPARISON_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "results",
    "model_comparison.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    possible_files = [
        ENGINEERED_DATA_PATH,
        CLEANED_DATA_PATH
    ]

    for file_path in possible_files:

        if os.path.exists(file_path):

            try:
                data = pd.read_csv(file_path)

                # Remove completely empty columns
                data = data.dropna(
                    axis=1,
                    how="all"
                )

                # Remove accidental index columns
                data = data.loc[
                    :,
                    ~data.columns.astype(str)
                    .str.startswith("Unnamed")
                ]

                return data

            except Exception as error:

                st.error(
                    f"Unable to read dataset: {error}"
                )

                return None

    return None


df = load_dataset()


# ============================================================
# DATASET CHECK
# ============================================================

if df is None:

    st.error(
        "Dataset not found."
    )

    st.info(
        "Make sure attendance_engineered.csv exists inside:"
    )

    st.code(
        "data/cleaned/attendance_engineered.csv"
    )

    st.stop()


TARGET = "Attendance %"


if TARGET not in df.columns:

    st.error(
        f"Target column '{TARGET}' is missing."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

total_records = len(df)

total_features = len(df.columns)

average_attendance = pd.to_numeric(
    df[TARGET],
    errors="coerce"
).mean()


# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def load_saved_model():

    if not os.path.exists(MODEL_PATH):
        return None

    # --------------------------------------------------------
    # First try joblib
    # --------------------------------------------------------

    try:

        model = joblib.load(
            MODEL_PATH
        )

        return model

    except Exception:
        pass


    # --------------------------------------------------------
    # Then try pickle
    # --------------------------------------------------------

    try:

        with open(
            MODEL_PATH,
            "rb"
        ) as file:

            model = pickle.load(
                file
            )

        return model

    except Exception:
        return None


saved_model = load_saved_model()


# ============================================================
# FALLBACK MODEL
# ============================================================

def prepare_fallback_data(data):

    temp = data.copy()


    # --------------------------------------------------------
    # Remove target/leakage columns
    # --------------------------------------------------------

    remove_columns = [
        "Attendance %",
        "Students Present",
        "Attendance Category",
        "Lecture Occurred"
    ]

    X = temp.drop(
        columns=remove_columns,
        errors="ignore"
    )

    y = pd.to_numeric(
        temp[TARGET],
        errors="coerce"
    )


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if "Date" in X.columns:

        date_values = pd.to_datetime(
            X["Date"],
            errors="coerce"
        )

        X["Date_Year"] = (
            date_values.dt.year
        )

        X["Date_Month"] = (
            date_values.dt.month
        )

        X["Date_Day"] = (
            date_values.dt.day
        )

        X["Date_DayOfWeek"] = (
            date_values.dt.dayofweek
        )

        X = X.drop(
            columns=["Date"]
        )


    # --------------------------------------------------------
    # START TIME
    # --------------------------------------------------------

    if "Start Time" in X.columns:

        time_values = pd.to_datetime(
            X["Start Time"].astype(str),
            errors="coerce"
        )

        X["Start_Hour_From_Time"] = (
            time_values.dt.hour
        )

        X["Start_Minute"] = (
            time_values.dt.minute
        )

        X = X.drop(
            columns=["Start Time"]
        )


    # --------------------------------------------------------
    # BOOLEAN COLUMNS
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].dtype == bool:

            X[column] = (
                X[column].astype(int)
            )


    # --------------------------------------------------------
    # REMOVE INVALID TARGET ROWS
    # --------------------------------------------------------

    valid = y.notna()

    X = X.loc[
        valid
    ].copy()

    y = y.loc[
        valid
    ].copy()

    return X, y


@st.cache_resource
def train_fallback_model(data):

    X, y = prepare_fallback_data(
        data
    )


    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns.tolist()
    )


    numerical_features = (
        X.select_dtypes(
            include=[
                "int64",
                "int32",
                "int16",
                "float64",
                "float32"
            ]
        ).columns.tolist()
    )


    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )


    model.fit(
        X,
        y
    )

    return model


# ============================================================
# SELECT MODEL
# ============================================================

if saved_model is not None:

    model = saved_model

    model_source = (
        "Saved model: best_model.pkl"
    )

else:

    model = train_fallback_model(
        df
    )

    model_source = (
        "Fallback Random Forest trained from dataset"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "# 📊 Classroom Attendance Prediction System"
)

st.markdown(
    """
    <div class="subtitle">
    A Machine Learning based system for predicting the expected
    attendance percentage of an upcoming lecture using historical
    attendance and academic schedule information.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    "## 🤖 Model Performance"
)


best_model_name = "Random Forest"

mae_value = 12.48

rmse_value = 14.24

r2_value = 0.23


# ------------------------------------------------------------
# Read model comparison
# ------------------------------------------------------------

if os.path.exists(
    MODEL_COMPARISON_PATH
):

    try:

        comparison_temp = pd.read_csv(
            MODEL_COMPARISON_PATH
        )

        if "Model" in comparison_temp.columns:

            rf_rows = comparison_temp[
                comparison_temp["Model"]
                .astype(str)
                .str.contains(
                    "Random Forest",
                    case=False,
                    na=False
                )
            ]

            if not rf_rows.empty:

                rf_row = rf_rows.iloc[0]

                if "MAE" in comparison_temp.columns:
                    mae_value = float(
                        rf_row["MAE"]
                    )

                if "RMSE" in comparison_temp.columns:
                    rmse_value = float(
                        rf_row["RMSE"]
                    )

                if "R2" in comparison_temp.columns:
                    r2_value = float(
                        rf_row["R2"]
                    )

                elif "R²" in comparison_temp.columns:
                    r2_value = float(
                        rf_row["R²"]
                    )

    except Exception:
        pass


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Best Model",
        best_model_name
    )


with m2:

    st.metric(
        "MAE",
        f"{mae_value:.2f}"
    )


with m3:

    st.metric(
        "RMSE",
        f"{rmse_value:.2f}"
    )


with m4:

    st.metric(
        "R² Score",
        f"{r2_value:.2f}"
    )


st.caption(
    f"Model source: {model_source}"
)


# ============================================================
# LECTURE INFORMATION
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    "## 📚 Lecture Information"
)


l1, l2, l3 = st.columns(3)


with l1:

    lecture_date = st.date_input(
        "Lecture Date"
    )

    selected_day = (
        lecture_date.strftime("%A")
    )

    st.info(
        f"Selected Day: {selected_day}"
    )


with l2:

    if "Start Time" in df.columns:

        time_values = (
            df["Start Time"]
            .dropna()
            .astype(str)
            .str.extract(
                r"(\d{1,2}:\d{2})"
            )[0]
            .dropna()
            .unique()
            .tolist()
        )

        time_values = sorted(
            time_values
        )

    else:

        time_values = [
            "08:30",
            "09:15",
            "10:15",
            "11:15",
            "13:30"
        ]


    if not time_values:

        time_values = [
            "08:30",
            "09:15",
            "10:15",
            "11:15",
            "13:30"
        ]


    start_time = st.selectbox(
        "Start Time",
        time_values
    )


with l3:

    if "Semester" in df.columns:

        semester_values = (
            pd.to_numeric(
                df["Semester"],
                errors="coerce"
            )
            .dropna()
            .unique()
            .tolist()
        )

    else:

        semester_values = [3]


    semester_default = int(
        semester_values[0]
    )


    semester = st.number_input(
        "Semester",
        min_value=1,
        max_value=10,
        value=semester_default,
        step=1
    )


# ============================================================
# LECTURE SECOND ROW
# ============================================================

l4, l5, l6 = st.columns(3)


with l4:

    lecture_number = st.number_input(
        "Lecture Number",
        min_value=1,
        max_value=20,
        value=1,
        step=1
    )


with l5:

    if "Total Enrolled Students" in df.columns:

        enrolled = pd.to_numeric(
            df["Total Enrolled Students"],
            errors="coerce"
        ).dropna()

        if not enrolled.empty:

            default_students = int(
                enrolled.median()
            )

        else:

            default_students = 200

    else:

        default_students = 200


    total_students = st.number_input(
        "Total Enrolled Students",
        min_value=1,
        max_value=1000,
        value=default_students,
        step=1
    )


with l6:

    if "Branch" in df.columns:

        branch_values = sorted(
            df["Branch"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        branch_values = ["MCA"]


    branch = st.selectbox(
        "Branch",
        branch_values
    )


# ============================================================
# LECTURE THIRD ROW
# ============================================================

l7, l8, l9 = st.columns(3)


with l7:

    if "Section" in df.columns:

        section_values = sorted(
            df["Section"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        section_values = ["A+B"]


    section = st.selectbox(
        "Section",
        section_values
    )


with l8:

    if "Classroom" in df.columns:

        classroom_values = sorted(
            df["Classroom"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        classroom_values = [
            "401",
            "403"
        ]


    classroom = st.selectbox(
        "Classroom",
        classroom_values
    )


with l9:

    if "Faculty ID" in df.columns:

        faculty_values = sorted(
            df["Faculty ID"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        faculty_values = [
            "F_001"
        ]


    faculty_id = st.selectbox(
        "Faculty ID",
        faculty_values
    )


# ============================================================
# ACADEMIC INFORMATION
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    "## 🎓 Academic Information"
)


a1, a2, a3 = st.columns(3)


with a1:

    if "Subject" in df.columns:

        subject_values = sorted(
            df["Subject"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        subject_values = [
            "Mobile Application Development"
        ]


    subject = st.selectbox(
        "Subject",
        subject_values
    )


with a2:

    if "Practical/Theory" in df.columns:

        practical_values = sorted(
            df["Practical/Theory"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        practical_values = [
            "Theory",
            "Practical"
        ]


    practical_theory = st.selectbox(
        "Practical / Theory",
        practical_values
    )


with a3:

    if "Weather" in df.columns:

        weather_values = sorted(
            df["Weather"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        weather_values = [
            "Sunny",
            "Cloudy",
            "Rainy"
        ]


    weather = st.selectbox(
        "Weather",
        weather_values
    )


# ============================================================
# ACADEMIC ROW 2
# ============================================================

a4, a5, a6 = st.columns(3)


with a4:

    if "Test Week" in df.columns:

        test_values = sorted(
            df["Test Week"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        test_values = [
            "No",
            "Yes"
        ]


    test_week = st.selectbox(
        "Test Week",
        test_values
    )


with a5:

    if "Assignment Due" in df.columns:

        assignment_values = sorted(
            df["Assignment Due"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        assignment_values = [
            "No",
            "Yes"
        ]


    assignment_due = st.selectbox(
        "Assignment Due",
        assignment_values
    )


with a6:

    if "Holiday" in df.columns:

        holiday_values = sorted(
            df["Holiday"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        holiday_values = [
            "No",
            "Yes"
        ]


    holiday = st.selectbox(
        "Holiday",
        holiday_values
    )


# ============================================================
# ACADEMIC ROW 3
# ============================================================

a7, a8, a9 = st.columns(3)


with a7:

    if "Special Event" in df.columns:

        event_values = sorted(
            df["Special Event"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        event_values = [
            "No",
            "Yes"
        ]


    special_event = st.selectbox(
        "Special Event",
        event_values
    )


with a8:

    if "Faculty Experience Years" in df.columns:

        experience = pd.to_numeric(
            df["Faculty Experience Years"],
            errors="coerce"
        ).dropna()

        if not experience.empty:

            default_experience = int(
                experience.median()
            )

        else:

            default_experience = 5

    else:

        default_experience = 5


    faculty_experience = st.number_input(
        "Faculty Experience (Years)",
        min_value=0,
        max_value=50,
        value=default_experience,
        step=1
    )


with a9:

    previous_attendance = st.number_input(
        "Previous Lecture Attendance %",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


# ============================================================
# HISTORICAL ATTENDANCE
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    "## 📈 Historical Attendance"
)


h1, h2, h3 = st.columns(3)


with h1:

    previous_3_average = st.number_input(
        "Previous 3 Lecture Average %",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


with h2:

    if "Previous Attendance Missing" in df.columns:

        missing_values = sorted(
            df["Previous Attendance Missing"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        missing_values = [
            "No",
            "Yes"
        ]


    previous_missing = st.selectbox(
        "Previous Attendance Missing",
        missing_values
    )


with h3:

    if "Week Number" in df.columns:

        weeks = pd.to_numeric(
            df["Week Number"],
            errors="coerce"
        ).dropna()

        if not weeks.empty:

            default_week = int(
                weeks.median()
            )

        else:

            default_week = 5

    else:

        default_week = 5


    week_number = st.number_input(
        "Week Number",
        min_value=1,
        max_value=52,
        value=default_week,
        step=1
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_input_dataframe():

    # --------------------------------------------------------
    # START HOUR
    # --------------------------------------------------------

    try:

        hour = int(
            str(start_time).split(":")[0]
        )

    except Exception:

        hour = 8


    # --------------------------------------------------------
    # DAY OF SEMESTER
    # --------------------------------------------------------

    day_of_semester = (
        lecture_date.timetuple().tm_yday
    )


    # --------------------------------------------------------
    # INPUT DICTIONARY
    # --------------------------------------------------------

    input_data = {}


    for column in df.columns:

        if column == TARGET:

            continue


        # ----------------------------------------------------
        # Basic columns
        # ----------------------------------------------------

        if column == "Date":

            input_data[column] = str(
                lecture_date
            )


        elif column == "Day":

            input_data[column] = (
                selected_day
            )


        elif column == "Lecture Number":

            input_data[column] = (
                lecture_number
            )


        elif column == "Start Time":

            input_data[column] = (
                start_time
            )


        elif column == "Subject":

            input_data[column] = (
                subject
            )


        elif column == "Faculty ID":

            input_data[column] = (
                faculty_id
            )


        elif column == "Semester":

            input_data[column] = (
                semester
            )


        elif column == "Branch":

            input_data[column] = (
                branch
            )


        elif column == "Section":

            input_data[column] = (
                section
            )


        elif column == "Classroom":

            input_data[column] = (
                classroom
            )


        elif column == "Total Enrolled Students":

            input_data[column] = (
                total_students
            )


        elif column == "Previous Lecture Attendance":

            input_data[column] = (
                previous_attendance
            )


        elif column == "Practical/Theory":

            input_data[column] = (
                practical_theory
            )


        elif column == "Test Week":

            input_data[column] = (
                test_week
            )


        elif column == "Assignment Due":

            input_data[column] = (
                assignment_due
            )


        elif column == "Holiday":

            input_data[column] = (
                holiday
            )


        elif column == "Weather":

            input_data[column] = (
                weather
            )


        elif column == "Special Event":

            input_data[column] = (
                special_event
            )


        elif column == "Faculty Experience":

            input_data[column] = (
                f"{faculty_experience} years"
            )


        elif column == "Faculty Experience Years":

            input_data[column] = (
                faculty_experience
            )


        elif column == "Previous Attendance Missing":

            input_data[column] = (
                previous_missing
            )


        # ====================================================
        # IMPORTANT ENGINEERED FEATURES
        # ====================================================

        elif column == "Start Hour":

            input_data[column] = (
                hour
            )


        elif column == "Day of Semester":

            input_data[column] = (
                day_of_semester
            )


        elif column == "Week Number":

            input_data[column] = (
                week_number
            )


        elif column == "Rolling 3 Lecture Attendance":

            input_data[column] = (
                previous_3_average
            )


        # ----------------------------------------------------
        # Other columns
        # ----------------------------------------------------

        else:

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                value = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).median()

                if pd.isna(value):

                    value = 0

                input_data[column] = value

            else:

                modes = (
                    df[column]
                    .dropna()
                    .mode()
                )

                if not modes.empty:

                    input_data[column] = (
                        modes.iloc[0]
                    )

                else:

                    input_data[column] = ""


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )


    # ========================================================
    # FORCE ENGINEERED COLUMNS
    # ========================================================

    # This specifically fixes your Start Hour error.

    input_df["Start Hour"] = hour

    input_df["Day of Semester"] = (
        day_of_semester
    )

    input_df["Week Number"] = (
        week_number
    )

    input_df["Rolling 3 Lecture Attendance"] = (
        previous_3_average
    )


    return input_df


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)


predict_clicked = st.button(
    "🔮 Predict Attendance"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    try:

        # ----------------------------------------------------
        # Create input
        # ----------------------------------------------------

        input_df = (
            create_input_dataframe()
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # If model has feature_names_in_,
        # make sure ALL required columns exist.
        # ----------------------------------------------------

        if hasattr(
            model,
            "feature_names_in_"
        ):

            required_features = list(
                model.feature_names_in_
            )


            for feature in required_features:

                if feature not in input_df.columns:

                    # Engineered features

                    if feature == "Start Hour":

                        input_df[feature] = int(
                            str(start_time)
                            .split(":")[0]
                        )


                    elif feature == "Day of Semester":

                        input_df[feature] = (
                            lecture_date
                            .timetuple()
                            .tm_yday
                        )


                    elif feature == "Week Number":

                        input_df[feature] = (
                            week_number
                        )


                    elif feature == "Rolling 3 Lecture Attendance":

                        input_df[feature] = (
                            previous_3_average
                        )


                    # Dataset feature

                    elif feature in df.columns:

                        if pd.api.types.is_numeric_dtype(
                            df[feature]
                        ):

                            value = pd.to_numeric(
                                df[feature],
                                errors="coerce"
                            ).median()

                            if pd.isna(value):

                                value = 0

                            input_df[feature] = (
                                value
                            )

                        else:

                            modes = (
                                df[feature]
                                .dropna()
                                .mode()
                            )

                            if not modes.empty:

                                input_df[feature] = (
                                    modes.iloc[0]
                                )

                            else:

                                input_df[feature] = ""


                    else:

                        input_df[feature] = 0


            # ------------------------------------------------
            # Keep model feature order
            # ------------------------------------------------

            input_df = input_df[
                required_features
            ]


        # ====================================================
        # PIPELINE MODEL
        # ====================================================

        prediction = model.predict(
            input_df
        )


        predicted_attendance = float(
            prediction[0]
        )


        # ----------------------------------------------------
        # Keep prediction between 0 and 100
        # ----------------------------------------------------

        predicted_attendance = max(
            0,
            min(
                100,
                predicted_attendance
            )
        )


        # ====================================================
        # EXPECTED STUDENTS
        # ====================================================

        expected_present = round(
            total_students
            * predicted_attendance
            / 100
        )


        expected_absent = (
            total_students
            - expected_present
        )


        # ====================================================
        # CATEGORY
        # ====================================================

        if predicted_attendance < 50:

            category = (
                "Low Attendance"
            )

            message = (
                "The predicted attendance is below "
                "50%. Low student participation is expected."
            )

        elif predicted_attendance < 75:

            category = (
                "Medium Attendance"
            )

            message = (
                "The predicted attendance is between "
                "50% and 75%. Moderate student participation "
                "is expected."
            )

        else:

            category = (
                "High Attendance"
            )

            message = (
                "The predicted attendance is 75% or above. "
                "High student participation is expected."
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="divider"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "## 🎯 Prediction Result"
        )


        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "Predicted Attendance",
                f"{predicted_attendance:.2f}%"
            )


        with r2:

            st.metric(
                "Expected Students Present",
                expected_present
            )


        with r3:

            st.metric(
                "Expected Students Absent",
                expected_absent
            )


        # ----------------------------------------------------
        # Attendance category
        # ----------------------------------------------------

        if predicted_attendance < 50:

            st.error(
                f"🔴 {category}"
            )

        elif predicted_attendance < 75:

            st.warning(
                f"🟡 {category}"
            )

        else:

            st.success(
                f"🟢 {category}"
            )


        st.info(
            message
        )


        # ====================================================
        # STUDENT DISTRIBUTION
        # ====================================================

        st.markdown(
            "### 📊 Expected Student Distribution"
        )


        chart_data = pd.DataFrame(
            {
                "Status": [
                    "Present",
                    "Absent"
                ],
                "Students": [
                    expected_present,
                    expected_absent
                ]
            }
        )


        fig, ax = plt.subplots(
            figsize=(9, 4)
        )


        ax.bar(
            chart_data["Status"],
            chart_data["Students"]
        )


        ax.set_title(
            "Expected Students Present vs Absent"
        )


        ax.set_xlabel(
            "Attendance Status"
        )


        ax.set_ylabel(
            "Number of Students"
        )


        st.pyplot(
            fig,
            clear_figure=True
        )


    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )

        st.warning(
            "The model and input features do not have "
            "the same structure."
        )

        st.info(
            "If this message appears again, the exact "
            "features expected by best_model.pkl need "
            "to be matched with the application input."
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    "## 📁 Dataset Information"
)


d1, d2, d3 = st.columns(3)


with d1:

    st.metric(
        "Total Records",
        total_records
    )


with d2:

    st.metric(
        "Total Features",
        total_features
    )


with d3:

    st.metric(
        "Average Attendance",
        f"{average_attendance:.2f}%"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    "## 🤖 Model Comparison"
)


if os.path.exists(
    MODEL_COMPARISON_PATH
):

    try:

        comparison = pd.read_csv(
            MODEL_COMPARISON_PATH
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:

        st.warning(
            f"Could not load model comparison: {error}"
        )

else:

    st.info(
        "Model comparison file not found."
    )


# ============================================================
# ATTENDANCE ANALYTICS
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    "## 📊 Attendance Analytics"
)

st.write(
    "Explore historical attendance patterns from the dataset."
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Distribution",
        "📚 Subject Analysis",
        "📅 Schedule Analysis",
        "👨‍🏫 Faculty Analysis"
    ]
)


# ============================================================
# DISTRIBUTION
# ============================================================

with tab1:

    st.markdown(
        "### Attendance Percentage Distribution"
    )


    attendance_values = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    ).dropna()


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.hist(
        attendance_values,
        bins=10
    )


    ax.set_title(
        "Attendance Percentage Distribution"
    )


    ax.set_xlabel(
        "Attendance (%)"
    )


    ax.set_ylabel(
        "Number of Lectures"
    )


    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# SUBJECT ANALYSIS
# ============================================================

with tab2:

    if "Subject" in df.columns:

        subject_analysis = (
            df.groupby("Subject")[TARGET]
            .mean()
            .sort_values(
                ascending=False
            )
        )


        st.markdown(
            "### Average Attendance by Subject"
        )


        fig, ax = plt.subplots(
            figsize=(12, 6)
        )


        subject_analysis.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Attendance by Subject"
        )


        ax.set_xlabel(
            "Subject"
        )


        ax.set_ylabel(
            "Average Attendance (%)"
        )


        plt.xticks(
            rotation=45,
            ha="right"
        )


        st.pyplot(
            fig,
            clear_figure=True
        )


# ============================================================
# SCHEDULE ANALYSIS
# ============================================================

with tab3:

    schedule_df = df.copy()


    if "Start Hour" in schedule_df.columns:

        schedule_df["Hour"] = pd.to_numeric(
            schedule_df["Start Hour"],
            errors="coerce"
        )


    elif "Start Time" in schedule_df.columns:

        schedule_df["Hour"] = pd.to_datetime(
            schedule_df["Start Time"],
            errors="coerce"
        ).dt.hour


    else:

        schedule_df["Hour"] = np.nan


    schedule_analysis = (
        schedule_df.dropna(
            subset=["Hour"]
        )
        .groupby("Hour")[TARGET]
        .mean()
        .sort_index()
    )


    if not schedule_analysis.empty:

        st.markdown(
            "### Average Attendance by Start Hour"
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        schedule_analysis.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Attendance by Start Hour"
        )


        ax.set_xlabel(
            "Start Hour"
        )


        ax.set_ylabel(
            "Average Attendance (%)"
        )


        st.pyplot(
            fig,
            clear_figure=True
        )

    else:

        st.info(
            "Start Hour information is not available."
        )


# ============================================================
# FACULTY ANALYSIS
# ============================================================

with tab4:

    if "Faculty ID" in df.columns:

        faculty_analysis = (
            df.groupby("Faculty ID")[TARGET]
            .mean()
            .sort_values(
                ascending=False
            )
        )


        st.markdown(
            "### Average Attendance by Faculty"
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        faculty_analysis.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            "Average Attendance by Faculty"
        )


        ax.set_xlabel(
            "Faculty ID"
        )


        ax.set_ylabel(
            "Average Attendance (%)"
        )


        st.pyplot(
            fig,
            clear_figure=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.caption(
    "Classroom Attendance Prediction System | "
    "Machine Learning Based DSML Project"
)