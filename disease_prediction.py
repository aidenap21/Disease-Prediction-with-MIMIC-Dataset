
import pandas as pd
from pathlib import Path
from joblib import dump, load
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class DiseasePredictor:
    def __init__(self, random_state=21):
        self.model = None
        self.encoder = None
        self.feature_order = None
        self.data = None
        self.random_state = random_state

        self.script_dir = Path.cwd().parent.parent
        self.mimic_iv_dir = self.script_dir / 'physionet.org' / 'files' / 'mimiciv' / '3.1'


    def load_data(self):
        """Processes the MIMIC-IV dataset to create a DataFrame with vitals and diagnoses for each patient admission."""
        # Checks if data has already been processed and can be loaded
        if Path('disease_prediction_data.parquet').exists():
            self.data = pd.read_parquet('disease_prediction_data.parquet')
            return

        # Loads CSV files
        patients = pd.read_csv(self.mimic_iv_dir / 'hosp' / 'patients.csv')
        admissions = pd.read_csv(self.mimic_iv_dir / 'hosp' / 'admissions.csv')
        diagnoses = pd.read_csv(self.mimic_iv_dir / 'hosp' / 'diagnoses_icd.csv')
        diagnosis_labels = pd.read_csv(self.mimic_iv_dir / 'hosp' / 'd_icd_diagnoses.csv')

        # Map of IDs to vital names
        vital_ids = {
            220045: 'heart_rate',
            220210: 'respiratory_rate',
            223761: 'temperature',
            220277: 'spo2',
            220179: 'systolic_bp',
            220180: 'diastolic_bp',
            220181: 'mean_arterial_pressure',
        }

        # Load chartevent in chunks to avoid memory issues
        chunks = []
        for chunk in tqdm(pd.read_csv(
            self.mimic_iv_dir / 'icu' / 'chartevents.csv',
            usecols=['subject_id','hadm_id','itemid','valuenum'],
            chunksize=1_000_000
        ), desc='Processing chartevents'):
            
            # Filter by data that is relevant
            filtered = chunk[chunk['itemid'].isin(vital_ids.keys())]
            chunks.append(filtered)

        # Combine the chunks
        chartevents = pd.concat(chunks, ignore_index=True)

        # Merge diagnoses with labels to get names
        diagnoses = diagnoses.merge(
            diagnosis_labels,
            on='icd_code',
            how='left'
        )

        # Keep only the first diagnosis for each admission
        diagnoses = diagnoses[diagnoses['seq_num'] == 1]

        # Calculate mean, max, and min for each vital for each admission
        vitals_stats = chartevents.groupby(['hadm_id','itemid'])['valuenum'].agg(
            ['mean','max','min']
        ).unstack()

        # Rename columns to use vital names rather than IDs
        vitals_stats = vitals_stats.rename(columns=vital_ids, level=0)

        # Create new column names in the format "vital_statistic"
        vitals_stats.columns = [
            f'{vital_ids[itemid]}_{stat}' for stat, itemid in vitals_stats.columns
        ]

        # Merge vitals with diagnoses
        data = vitals_stats.merge(
            diagnoses[['hadm_id', 'long_title']],
            on='hadm_id'
        )

        # Merge patients with admissions to get age at admission
        admissions = admissions.merge(patients, on='subject_id')

        # Merge age information into the main dataset to have age feature
        data = data.merge(
            admissions[['hadm_id','anchor_age']],
            on='hadm_id'
        )

        # Rename age column and drop rows with missing values
        data = data.rename(columns={'anchor_age':'age'})
        data = data.dropna()

        # Keep only the top 10 most common diagnoses to reduce complexity of prediction
        top_diagnoses = data['long_title'].value_counts().nlargest(10).index
        data = data[data['long_title'].isin(top_diagnoses)]

        # Save data for future use
        data.to_parquet('disease_prediction_data.parquet')
        self.data = data

    def get_split_data(self, test_size=0.2):
        """Gets training and test data."""
        # Separate features and target variable
        X = self.data.drop(['long_title', 'hadm_id'], axis=1)
        y = self.data['long_title']

        # Load feature order and label encoder if they exist, otherwise create them
        if Path('feature_order.joblib').exists():
            self.feature_order = load('feature_order.joblib')
        else:
            self.feature_order = X.columns.tolist()

        if Path('label_encoder.joblib').exists():
            self.encoder = load('label_encoder.joblib')
        else:
            self.encoder = LabelEncoder()

        # Encode target variable
        y_encoded = self.encoder.fit_transform(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def load_model(self, X_train, y_train):
        """Loads a saved model or trains a new one."""
        # Load trained model if it exists, otherwise train a new one
        if Path('disease_prediction_model.joblib').exists():
            self.model = load('disease_prediction_model.joblib')
            return
        self.train_model(X_train, y_train)

    def train_model(
            self,
            X_train,
            y_train,
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=3,
            max_features='log2',
            class_weight='balanced',
        ):
        """Trains a Random Forest Classifier on the training data and saves it."""
        # Create a Random Forest Classifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=self.random_state,
        )

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Save the trained model, label encoder, and feature order for future use
        dump(model, 'disease_prediction_model.joblib')
        dump(self.encoder, 'label_encoder.joblib')
        dump(self.feature_order, 'feature_order.joblib')

        self.model = model

    def evaluate_model(self, X_test, y_test):
        """Evaluates the model's performance on the test set."""
        # Make predictions on the test set
        predictions = self.model.predict(X_test)

        # Evaluate the model's performance
        print('Accuracy:', accuracy_score(y_test, predictions))

        print('Classification Report:')
        print(classification_report(y_test, predictions))

        # Print feature importance to show which features impact the model's predictions the most
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_order
        ).sort_values(ascending=False)

        print(importance.head(10))

    def predict_patient(self, vitals_dict):
        """Predicts the disease for a patient based on their vitals."""
        # Create DataFrame for the vitals given
        patient_df = pd.DataFrame([vitals_dict])
        patient_df = patient_df[self.feature_order]

        # Predict the disease with the given vitals
        prediction = self.model.predict(patient_df)

        # Convert the predicted label back to the original disease name
        disease = self.encoder.inverse_transform(prediction)[0]
        return disease

def generate_patient():
    """Generates a random patient with vitals in realistic ranges."""
    age = random.randint(18, 90)

    # Heart rate
    heart_rate_mean = random.randint(60, 130)
    heart_rate_min = heart_rate_mean - random.randint(5, 20)
    heart_rate_max = heart_rate_mean + random.randint(5, 25)

    # Respiratory rate
    respiratory_rate_mean = random.randint(12, 30)
    respiratory_rate_min = respiratory_rate_mean - random.randint(2, 6)
    respiratory_rate_max = respiratory_rate_mean + random.randint(2, 8)

    # Temperature
    temperature_mean = round(random.uniform(36.0, 39.5), 1)
    temperature_min = round(temperature_mean - random.uniform(0.2, 1.0), 1)
    temperature_max = round(temperature_mean + random.uniform(0.2, 1.2), 1)

    # SpO2
    spo2_mean = random.randint(88, 99)
    spo2_min = spo2_mean - random.randint(2, 8)
    spo2_max = min(100, spo2_mean + random.randint(1, 3))

    # Blood pressure
    systolic_bp_mean = random.randint(85, 140)
    systolic_bp_min = systolic_bp_mean - random.randint(5, 15)
    systolic_bp_max = systolic_bp_mean + random.randint(5, 15)

    diastolic_bp_mean = random.randint(50, 90)
    diastolic_bp_min = diastolic_bp_mean - random.randint(3, 10)
    diastolic_bp_max = diastolic_bp_mean + random.randint(3, 10)

    mean_arterial_pressure_mean = int((systolic_bp_mean + 2 * diastolic_bp_mean) / 3)
    mean_arterial_pressure_min = mean_arterial_pressure_mean - random.randint(3, 8)
    mean_arterial_pressure_max = mean_arterial_pressure_mean + random.randint(3, 8)

    patient = {
        'heart_rate_mean': heart_rate_mean,
        'heart_rate_max': heart_rate_max,
        'heart_rate_min': heart_rate_min,
        'respiratory_rate_mean': respiratory_rate_mean,
        'respiratory_rate_max': respiratory_rate_max,
        'respiratory_rate_min': respiratory_rate_min,
        'temperature_mean': temperature_mean,
        'temperature_max': temperature_max,
        'temperature_min': temperature_min,
        'spo2_mean': spo2_mean,
        'spo2_max': spo2_max,
        'spo2_min': spo2_min,
        'systolic_bp_mean': systolic_bp_mean,
        'systolic_bp_max': systolic_bp_max,
        'systolic_bp_min': systolic_bp_min,
        'diastolic_bp_mean': diastolic_bp_mean,
        'diastolic_bp_max': diastolic_bp_max,
        'diastolic_bp_min': diastolic_bp_min,
        'mean_arterial_pressure_mean': mean_arterial_pressure_mean,
        'mean_arterial_pressure_max': mean_arterial_pressure_max,
        'mean_arterial_pressure_min': mean_arterial_pressure_min,
        'age': age
    }

    return patient

if __name__ == '__main__':
    predictor = DiseasePredictor()
    predictor.load_data()
    X_train, X_test, y_train, y_test = predictor.get_split_data()
    predictor.load_model(X_train, y_train)
    predictor.evaluate_model(X_test, y_test)

    # Example predictions
    patients = [
        {
            'heart_rate_mean': 110,
            'heart_rate_max': 125,
            'heart_rate_min': 95,
            'respiratory_rate_mean': 24,
            'respiratory_rate_max': 30,
            'respiratory_rate_min': 18,
            'temperature_mean': 38.2,
            'temperature_max': 39.0,
            'temperature_min': 37.5,
            'spo2_mean': 90,
            'spo2_max': 94,
            'spo2_min': 85,
            'systolic_bp_mean': 95,
            'systolic_bp_max': 105,
            'systolic_bp_min': 85,
            'diastolic_bp_mean': 60,
            'diastolic_bp_max': 70,
            'diastolic_bp_min': 50,
            'mean_arterial_pressure_mean': 72,
            'mean_arterial_pressure_max': 80,
            'mean_arterial_pressure_min': 65,
            'age': 65
        },
        {
            'heart_rate_mean': 118,
            'heart_rate_max': 135,
            'heart_rate_min': 100,
            'respiratory_rate_mean': 26,
            'respiratory_rate_max': 32,
            'respiratory_rate_min': 22,
            'temperature_mean': 38.7,
            'temperature_max': 39.5,
            'temperature_min': 38.0,
            'spo2_mean': 92,
            'spo2_max': 95,
            'spo2_min': 88,
            'systolic_bp_mean': 88,
            'systolic_bp_max': 95,
            'systolic_bp_min': 78,
            'diastolic_bp_mean': 55,
            'diastolic_bp_max': 60,
            'diastolic_bp_min': 48,
            'mean_arterial_pressure_mean': 66,
            'mean_arterial_pressure_max': 70,
            'mean_arterial_pressure_min': 60,
            'age': 72
        },
        {
            'heart_rate_mean': 95,
            'heart_rate_max': 110,
            'heart_rate_min': 80,
            'respiratory_rate_mean': 20,
            'respiratory_rate_max': 24,
            'respiratory_rate_min': 18,
            'temperature_mean': 37.8,
            'temperature_max': 38.3,
            'temperature_min': 37.3,
            'spo2_mean': 94,
            'spo2_max': 97,
            'spo2_min': 91,
            'systolic_bp_mean': 115,
            'systolic_bp_max': 125,
            'systolic_bp_min': 105,
            'diastolic_bp_mean': 70,
            'diastolic_bp_max': 75,
            'diastolic_bp_min': 65,
            'mean_arterial_pressure_mean': 85,
            'mean_arterial_pressure_max': 90,
            'mean_arterial_pressure_min': 80,
            'age': 78
        },
    ]

    print()
    for i, patient in enumerate(patients):
        print(f'Patient {i}:', predictor.predict_patient(patient))

    print()
    # Generate random patients
    for i in range(5):
        random_patient = generate_patient()
        print(f'Random Patient {i}:', predictor.predict_patient(random_patient))


