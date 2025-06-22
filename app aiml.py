import streamlit as st
import os
import cv2
import numpy as np
import pkg_resources
import warnings
import pandas as pd
from datetime import datetime

# Check protobuf version
protobuf_version = pkg_resources.get_distribution('protobuf').version
if not protobuf_version.startswith('3.20'):
    warnings.warn(f"Current protobuf version is {protobuf_version}. Version 3.20.x is recommended.")

# Import TensorFlow after version check
import tensorflow as tf
from src.data_processing.preprocessing import Preprocessor
from src.config import IMG_SIZE
from predict import get_model_path, verify_image_path
import plotly.graph_objects as go

def create_confidence_chart(predictions, class_names):
    """Create a horizontal bar chart for prediction confidences"""
    fig = go.Figure(go.Bar(
        x=[float(pred) for pred in predictions],
        y=class_names,
        orientation='h',
        text=[f'{float(pred):.2%}' for pred in predictions],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Prediction Confidence by Class',
        xaxis_title='Confidence',
        yaxis_title='Class',
        height=400
    )
    
    return fig

def create_report(patient_data, predicted_class, confidence, top_3_predictions, class_names):
    """Generate a clinical report based on predictions and patient data"""
    
    # Risk assessment based on clinical data
    risk_factors = []
    if patient_data["WBC_count"] > 50000:
        risk_factors.append("Elevated WBC count")
    if patient_data["WBC_count"] < 4000:
        risk_factors.append("Low WBC count")
    if patient_data["RBC_count"] < 4.0:
        risk_factors.append("Low RBC count")
    if patient_data["hemoglobin"] < 10:
        risk_factors.append("Low hemoglobin")
    if patient_data["platelet_count"] < 100000:
        risk_factors.append("Low platelet count")
    if patient_data["LDH"] > 250:
        risk_factors.append("Elevated LDH")
    if patient_data["blast_percentage"] > 20:
        risk_factors.append("High blast percentage")
    if patient_data["ESR"] > 50:
        risk_factors.append("Elevated ESR")

    # Add symptom-based risk factors
    symptoms = []
    if patient_data["bone_pain"]: symptoms.append("Bone pain")
    if patient_data["fatigue"]: symptoms.append("Fatigue")
    if patient_data["fever_history"]: symptoms.append("Fever/Infection history")
    if patient_data["weight_loss"]: symptoms.append("Weight loss")
    if patient_data["easy_bruising"]: symptoms.append("Easy bruising/bleeding")
    if patient_data["night_sweats"]: symptoms.append("Night sweats")

    # Define cancer cell types
    cancer_cell_types = ['BLA', 'FGC', 'HAC', 'LYT', 'MYB', 'PMO', 'PEB', 'PLM', 'ABE']
    developmental_cell_types = ['EBO', 'MMZ']
    technical_classification = ['ART', 'KSC', 'NIF', 'OTH']
    normal_cell_types = ['EOS', 'BAS', 'MON', 'LYM', 'NGS', 'NGB']
    
    # Enhanced class-specific recommendations with severity levels
    class_recommendations = {
    'BLA': {
        'description': 'Blast Cells - Primary indicator of acute leukemia',
        'high_risk': """🔴 CRITICAL: IMMEDIATE HEMATOLOGIC EMERGENCY
        
        Primary Actions (Within 24 Hours):
        1. STAT Hematology consultation
        2. Emergency bone marrow biopsy/aspirate
        3. Urgent molecular testing panel:
           - NPM1, FLT3-ITD/TKD mutations
           - BCR-ABL1 fusion (Ph chromosome)
           - CEBPA, RUNX1, TP53 mutations
        
        Critical Testing:
        • Complete blood count with manual differential
        • DIC panel (PT, APTT, fibrinogen, D-dimer)
        • Tumor lysis monitoring (uric acid, LDH, K, Ca, P)
        • HLA typing for potential transplant
        
        Immediate Precautions:
        • Infection prophylaxis
        • Bleeding precautions
        • Daily metabolic monitoring
        • Transfusion support as needed""",
        
        'low_risk': """🟡 URGENT: 24-48 Hour Hematology Evaluation
        
        Required Testing:
        1. Bone marrow evaluation
        2. Flow cytometry
        3. Cytogenetic analysis
        
        Monitoring:
        • Twice weekly CBC
        • Infection surveillance
        • Bleeding precautions"""
    },
    
    'FGC': {
        'description': 'Faggot Cells - Pathognomonic for Acute Promyelocytic Leukemia (APL)',
        'high_risk': """🔴 CRITICAL: APL EMERGENCY - HIGH MORTALITY RISK
        
        Immediate Actions (Within Hours):
        1. STAT PML-RARA testing
        2. Begin ATRA immediately
        3. Urgent coagulation monitoring
        4. Consider arsenic trioxide (ATO)
        
        Critical Testing:
        • q4h coagulation parameters
        • Fibrinogen level q6h
        • Platelet count q6h
        • DIC monitoring
        
        Special Precautions:
        • Hemorrhage prevention protocol
        • Differentiation syndrome monitoring
        • No lumbar puncture/invasive procedures
        • Strict vital sign monitoring""",
        
        'low_risk': """🔴 Still URGENT - APL Protocol Required
        
        Required Actions:
        1. Immediate hematology consultation
        2. ATRA initiation
        3. Coagulation monitoring
        
        Monitoring:
        • Daily coagulation studies
        • Twice daily CBC
        • Daily electrolytes"""
    },
    
    'HAC': {
        'description': 'Hairy Cells - Indicative of Hairy Cell Leukemia',
        'high_risk': """🔴 URGENT: Evaluate for Hairy Cell Leukemia
        
        Primary Actions:
        1. BRAF V600E mutation testing
        2. Bone marrow biopsy with immunophenotyping
        3. Flow cytometry (CD11c, CD25, CD103)
        
        Additional Testing:
        • CT scan for splenomegaly
        • Tartrate-resistant acid phosphatase
        • Immunoglobulin levels
        
        Monitoring:
        • Infection surveillance
        • Spleen size assessment
        • Weekly blood counts""",
        
        'low_risk': """🟡 Prompt Evaluation Needed
        
        Required Testing:
        1. Flow cytometry panel
        2. Peripheral blood immunophenotyping
        3. CBC with differential
        
        Follow-up:
        • Monthly monitoring
        • Infection prevention
        • Vaccination status review"""
    },
    
    'LYT': {
        'description': 'Lymphoblast - Immature lymphocyte, potential indicator of ALL',
        'high_risk': """🔴 CRITICAL: ACUTE LYMPHOBLASTIC LEUKEMIA WORKUP
        
        Primary Actions (Within 24 Hours):
        1. STAT Hematology/Oncology consultation
        2. Emergency bone marrow biopsy
        3. Flow cytometry with B/T-cell markers
        4. Cytogenetic & molecular studies:
           - BCR-ABL1, MLL rearrangements
           - ETV6-RUNX1, hyperdiploid/hypodiploid
        
        Critical Testing:
        • Lumbar puncture for CSF analysis
        • HLA typing for transplant potential
        • Immunophenotyping panel
        • Tumor lysis syndrome prophylaxis
        
        Immediate Precautions:
        • Infection isolation if neutropenic
        • Tumor lysis monitoring
        • CNS prophylaxis planning
        • Fertility preservation discussion""",
        
        'low_risk': """🟡 URGENT: 24-48 Hour Hematology Evaluation
        
        Required Testing:
        1. Peripheral blood flow cytometry
        2. Bone marrow aspirate/biopsy
        3. Minimal residual disease assessment planning
        
        Monitoring:
        • Twice weekly CBC
        • Lymph node assessment
        • Hepatosplenomegaly evaluation"""
    },
    
    'MYB': {
        'description': 'Myeloblast - Myeloid precursor, potential AML indicator',
        'high_risk': """🔴 CRITICAL: ACUTE MYELOID LEUKEMIA WORKUP
        
        Primary Actions (Within 24 Hours):
        1. STAT Hematology consultation
        2. Emergency bone marrow biopsy/aspirate
        3. Cytogenetic & molecular analysis:
           - Core binding factor abnormalities
           - FLT3, NPM1, CEBPα, IDH1/2, TP53
           - PML-RARA exclusion
        
        Critical Testing:
        • Complete blood count with differential
        • Coagulation studies (DIC screening)
        • Comprehensive metabolic panel
        • HLA typing for transplant evaluation
        
        Immediate Precautions:
        • Tumor lysis prevention protocol
        • Neutropenic precautions
        • Bleeding precautions
        • Central line consideration""",
        
        'low_risk': """🟡 URGENT: 24-48 Hour Hematology Evaluation
        
        Required Testing:
        1. Bone marrow evaluation
        2. Flow cytometry
        3. Molecular profiling
        
        Monitoring:
        • Daily CBC with differential
        • Electrolyte monitoring
        • Blast percentage tracking"""
    },
    
    'PMO': {
        'description': 'Promyelocytes - Potential indicator of APL or other myeloid disorders',
        'high_risk': """🔴 URGENT: Rule Out APL
        
        Primary Actions:
        1. STAT PML-RARA PCR
        2. Urgent coagulation studies
        3. Consider ATRA initiation
        
        Critical Testing:
        • DIC panel q6h
        • Flow cytometry
        • Cytogenetic analysis t(15;17)
        
        Monitoring:
        • Bleeding precautions
        • q6h vital signs
        • Daily metabolic panel""",
        
        'low_risk': """🟡 Expedited Evaluation
        
        Required Testing:
        1. PML-RARA testing
        2. Coagulation profile
        3. Flow cytometry
        
        Monitoring:
        • Daily CBC
        • Coagulation studies
        • Clinical assessment"""
    },
    
    'PEB': {
        'description': 'Proerythroblast - May indicate erythroleukemia (AML-M6)',
        'high_risk': """🔴 URGENT: Evaluate for Erythroleukemia
        
        Primary Actions:
        1. Bone marrow biopsy/aspirate
        2. Cytogenetic analysis
        3. Erythroid markers
        
        Essential Testing:
        • Flow cytometry
        • Molecular studies
        • Iron studies/B12/folate
        
        Monitoring:
        • Daily CBC
        • Transfusion requirements
        • Iron chelation assessment""",
        
        'low_risk': """🟡 Prompt Evaluation
        
        Required Testing:
        1. Complete blood count
        2. Reticulocyte count
        3. Iron studies
        
        Follow-up:
        • Weekly monitoring
        • Transfusion assessment
        • Bone marrow planning"""
    },
    
    'PLM': {
        'description': 'Plasma Cell - Antibody-producing cell, elevated in multiple myeloma',
        'high_risk': """🔴 URGENT: Multiple Myeloma Evaluation
        
        Primary Actions (Within 48 Hours):
        1. Hematology/Oncology consultation
        2. Bone marrow biopsy with immunohistochemistry
        3. Serum/urine protein electrophoresis
        4. Free light chain assay
        
        Critical Testing:
        • Comprehensive metabolic panel (Ca++, Cr)
        • Complete skeletal survey or low-dose CT
        • Beta-2 microglobulin
        • 24-hour urine protein
        
        Immediate Precautions:
        • Hypercalcemia monitoring
        • Renal function assessment
        • Bone pain management
        • Hyperviscosity screening""",
        
        'low_risk': """🟡 Expedited Evaluation
        
        Required Testing:
        1. Serum protein electrophoresis
        2. Immunofixation studies
        3. Quantitative immunoglobulins
        
        Monitoring:
        • Monthly CBC with differential
        • Renal function monitoring
        • Bone health assessment"""
    },
    
    'ABE': {
        'description': 'Abnormal Eosinophil - May indicate hypereosinophilic syndromes or myeloid neoplasms',
        'high_risk': """🔴 URGENT: Hypereosinophilic Evaluation
        
        Primary Actions:
        1. Hematology consultation
        2. Peripheral blood smear review
        3. Bone marrow examination with cytogenetics
        4. FIP1L1-PDGFRA/PDGFRB mutation testing
        
        Critical Testing:
        • Comprehensive organ function assessment
        • Cardiac evaluation (troponin, BNP, echo)
        • Pulmonary function tests
        • Serum tryptase levels
        
        Immediate Precautions:
        • Cardiac monitoring
        • Consider empiric steroid therapy
        • Thromboembolism prophylaxis
        • Organ damage assessment""",
        
        'low_risk': """🟡 Prompt Evaluation
        
        Required Testing:
        1. Complete blood count with differential
        2. Peripheral blood FISH for eosinophilia-associated mutations
        3. Allergy/parasite workup
        
        Monitoring:
        • Weekly CBC with differential
        • IgE levels
        • Stool examination for parasites"""
    }
}

    # Generate report with improved formatting
    report = f"""
    ## 🏥 Clinical Report
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

    ### 🔬 Image Analysis Results
    **Primary Classification:** {predicted_class} ({confidence:.2%})
    **Alternative Possibilities:** {', '.join([f"{class_names[idx]}" for idx in top_3_predictions[1:]])}

    ### 📊 Laboratory Values
    
    | Parameter          | Value                  | Reference Range    | Status                                                                                      |
    |:-------------------|:-----------------------|:-------------------|:--------------------------------------------------------------------------------------------|
    | WBC Count          | {patient_data['WBC_count']:,} cells/µL      | 4,500-11,000        | {'🔴 High' if patient_data['WBC_count'] > 11000 else '🔴 Low' if patient_data['WBC_count'] < 4500 else '✅ Normal'} |
    | RBC Count          | {patient_data['RBC_count']:.1f} M/µL        | 4.0-5.5             | {'🔴 High' if patient_data['RBC_count'] > 5.5 else '🔴 Low' if patient_data['RBC_count'] < 4.0 else '✅ Normal'}    |
    | Hemoglobin         | {patient_data['hemoglobin']:.1f} g/dL       | 12.0-16.0           | {'🔴 High' if patient_data['hemoglobin'] > 16 else '🔴 Low' if patient_data['hemoglobin'] < 12 else '✅ Normal'}    |
    | Platelets          | {patient_data['platelet_count']:,} K/µL     | 150,000-450,000     | {'🔴 High' if patient_data['platelet_count'] > 450000 else '🔴 Low' if patient_data['platelet_count'] < 150000 else '✅ Normal'} |
    | LDH                | {patient_data['LDH']} U/L                  | 100-220             | {'🔴 High' if patient_data['LDH'] > 220 else '🔴 Low' if patient_data['LDH'] < 100 else '✅ Normal'}               |
    | Blast Percentage   | {patient_data['blast_percentage']:.1f}%     | < 5%                | {'🔴 High' if patient_data['blast_percentage'] > 5 else '✅ Normal'}                                               |
    | ESR                | {patient_data['ESR']} mm/hr                | 0-20                | {'🔴 High' if patient_data['ESR'] > 20 else '✅ Normal'}                                                           |

    ### 🔍 Clinical Assessment
    **Reported Symptoms:**
    {chr(10).join(['• ' + symptom for symptom in symptoms]) if symptoms else '• No symptoms reported'}

    **Risk Factors:**
    {chr(10).join(['• ' + factor for factor in risk_factors]) if risk_factors else '• No significant risk factors identified'}

    ### 📋 Recommendation
    """

    # Add condition-specific recommendations
    high_risk = len(risk_factors) >= 2 or patient_data["blast_percentage"] > 20
    
    if predicted_class in cancer_cell_types:
        if high_risk:
            report += f"⚠️ **HIGH RISK MALIGNANT CELLS DETECTED**\n{class_recommendations[predicted_class]['high_risk']}"
        else:
            report += f"⚠️ **POTENTIAL MALIGNANT CELLS**\n{class_recommendations[predicted_class]['low_risk']}"
    elif predicted_class in normal_cell_types:
        report += """✅ **NORMAL CELL MORPHOLOGY**
        
        Recommended Actions:
        • Routine follow-up as clinically indicated
        • No immediate intervention required
        • Consider repeat CBC in 6 months"""
    else:
        report += """⚠️ **UNCERTAIN CLASSIFICATION**
        
        Recommended Actions:
        1. Repeat testing with fresh sample
        2. Manual differential review
        3. Consider flow cytometry
        4. Expert hematopathology consultation"""

    # Enhanced precautions section with more comprehensive guidelines
    report += "\n\n### 🚨 Additional Precautions\n"
    
    # General precautions based on lab values
    if patient_data["WBC_count"] < 3000:
        report += "\n#### 🛡️ Neutropenia Precautions:"
        report += "\n- Strict hand hygiene and visitor restrictions"
        report += "\n- HEPA-filtered room if ANC < 500"
        report += "\n- Avoid raw fruits, vegetables, and flowers"
        report += "\n- Consider prophylactic antibiotics if prolonged"
        report += "\n- Monitor temperature q4h"
        report += "\n- Prompt workup of fever > 38.0°C"
    
    if patient_data["platelet_count"] < 50000:
        report += "\n\n#### 🩸 Thrombocytopenia Precautions:"
        report += "\n- Avoid NSAIDs and anticoagulants"
        report += "\n- Minimize invasive procedures"
        report += "\n- Soft toothbrush and electric razor"
        report += "\n- Platelet transfusion if < 10,000 or bleeding"
        report += "\n- Avoidance of strenuous activity"
        report += "\n- Apply pressure to venipuncture sites for 5+ minutes"
    
    if patient_data["hemoglobin"] < 8:
        report += "\n\n#### ❤️ Anemia Management:"
        report += "\n- Consider RBC transfusion if symptomatic"
        report += "\n- Supplemental oxygen if hypoxemic"
        report += "\n- Iron, B12, folate studies"
        report += "\n- Limit physical exertion"
        report += "\n- Monitor for cardiac symptoms"
        report += "\n- Consider EPO if appropriate"
    
    # Symptom-based precautions
    if len(symptoms) >= 1:
        symptom_precautions = []
        
        if "fever_history" in symptoms:
            symptom_precautions.append("""
#### 🌡️ Fever Protocol:
- Blood cultures x2 before antibiotics
- Consider empiric broad-spectrum antibiotics
- Daily CBC with differential
- Chest radiograph
- Urinalysis and urine culture
- Isolation precautions pending culture results""")
            
        if "easy_bruising" in symptoms:
            symptom_precautions.append("""
#### 🩹 Bleeding Risk Management:
- Complete coagulation panel (PT, PTT, fibrinogen, D-dimer)
- Daily skin assessment
- Avoid IM injections
- Aminocaproic acid for mucosal bleeding
- Stool guaiac testing
- Consider recombinant Factor VIIa for severe bleeding""")
            
        if "fatigue" in symptoms:
            symptom_precautions.append("""
#### 😴 Fatigue Management:
- Functional assessment
- Energy conservation strategies
- Cardiac evaluation if severe
- Thyroid function tests
- Consider psychostimulants if severe
- Physical therapy consultation""")
            
        if "bone_pain" in symptoms:
            symptom_precautions.append("""
#### 🦴 Bone Pain Management:
- Radiographic skeletal survey
- Consider MRI of symptomatic areas
- Appropriate pain management protocol
- Bone density assessment
- Calcium and Vitamin D levels
- Consider bisphosphonate therapy""")
            
        if "weight_loss" in symptoms:
            symptom_precautions.append("""
#### ⚖️ Weight Loss Protocol:
- Nutritional assessment
- Caloric intake monitoring
- Consider TPN if severe
- Albumin and prealbumin monitoring
- Swallowing evaluation if indicated
- Weekly weight checks""")
            
        if "night_sweats" in symptoms:
            symptom_precautions.append("""
#### 💦 Night Sweats Management:
- Temperature monitoring
- Infectious disease consultation
- Consider TB testing
- Endocrine workup
- Moisture-wicking bedding
- Monitor for dehydration""")
        
        # Add symptom precautions to report
        for precaution in symptom_precautions:
            report += "\n" + precaution
    
    # Add general monitoring recommendations
    report += "\n\n### 📈 Ongoing Monitoring Plan\n"
    report += "- Schedule follow-up appointments based on risk assessment\n"
    report += "- Establish baseline for trend monitoring\n"
    report += "- Schedule periodic imaging as indicated\n"
    report += "- Consider genetic counseling if hereditary conditions suspected\n"
    report += "- Implement survivorship care plan if appropriate\n"
    report += "- Monitor quality of life metrics and adjust supportive care accordingly"

    return report

# Update the session state initialization with new parameters
def init_session_state():
    default_values = {
        # Basic Lab Parameters
        "WBC_count": 5000,
        "RBC_count": 4.5,
        "hemoglobin": 14.0,
        "platelet_count": 150000,
        "LDH": 200,
        "blast_percentage": 5.0,
        "ESR": 20,
        
        # Patient Symptoms
        "bone_pain": False,
        "fatigue": False,
        "fever_history": False,
        "weight_loss": False,
        "easy_bruising": False,
        "night_sweats": False
    }
    
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = default_values
    else:
        # Ensure all keys exist
        for key, value in default_values.items():
            if key not in st.session_state.patient_data:
                st.session_state.patient_data[key] = value

def main():
    st.set_page_config(page_title="Bone Marrow Cell Classifier", page_icon="🔬", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    st.title("Bone Marrow Cell Classification")
    st.markdown("""
    This application classifies bone marrow cell images into different categories.
    Upload an image to get started!
    """)

    # Enhanced Clinical data input form in sidebar
    st.sidebar.header("Clinical Data")
    with st.sidebar.form("clinical_data_form"):
        st.write("📊 Laboratory Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            wbc_count = st.number_input("WBC Count (cells/µL)", 0, 1000000, 
                                    value=st.session_state.patient_data["WBC_count"])
            rbc_count = st.number_input("RBC Count (M/µL)", 0.0, 10.0, 
                                    value=st.session_state.patient_data["RBC_count"])
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 
                                     value=st.session_state.patient_data["hemoglobin"])
            platelet_count = st.number_input("Platelet Count (K/µL)", 0, 1000000, 
                                         value=st.session_state.patient_data["platelet_count"])
        
        with col2:
            ldh = st.number_input("LDH (U/L)", 0, 1000, 
                               value=st.session_state.patient_data["LDH"])
            blast_percentage = st.number_input("Blast Percentage (%)", 0.0, 100.0, 
                                           value=st.session_state.patient_data["blast_percentage"])
            esr = st.number_input("ESR (mm/hr)", 0, 150, 
                               value=st.session_state.patient_data["ESR"])
        
        st.write("🔹 Patient Symptoms")
        col3, col4 = st.columns(2)
        
        with col3:
            bone_pain = st.checkbox("Bone Pain", 
                                value=st.session_state.patient_data["bone_pain"])
            fatigue = st.checkbox("Fatigue", 
                              value=st.session_state.patient_data["fatigue"])
            fever_history = st.checkbox("Fever/Infection History", 
                                    value=st.session_state.patient_data["fever_history"])
        
        with col4:
            weight_loss = st.checkbox("Weight Loss", 
                                  value=st.session_state.patient_data["weight_loss"])
            easy_bruising = st.checkbox("Easy Bruising/Bleeding", 
                                    value=st.session_state.patient_data["easy_bruising"])
            night_sweats = st.checkbox("Night Sweats", 
                                   value=st.session_state.patient_data["night_sweats"])
        
        if st.form_submit_button("Update Clinical Data"):
            st.session_state.patient_data.update({
                # Lab Parameters
                "WBC_count": wbc_count,
                "RBC_count": rbc_count,
                "hemoglobin": hemoglobin,
                "platelet_count": platelet_count,
                "LDH": ldh,
                "blast_percentage": blast_percentage,
                "ESR": esr,
                
                # Symptoms
                "bone_pain": bone_pain,
                "fatigue": fatigue,
                "fever_history": fever_history,
                "weight_loss": weight_loss,
                "easy_bruising": easy_bruising,
                "night_sweats": night_sweats
            })
    
    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.info("""
    This model can classify bone marrow cells into 21 different categories.
    Model: EfficientNetV2B0 + Random Forest
    """)
    
    # Class names
    class_names = ['ABE', 'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 
                   'HAC', 'KSC', 'LYI', 'LYT', 'MMZ', 'MON', 'MYB', 
                   'NGB', 'NGS', 'NIF', 'OTH', 'PEB', 'PLM', 'PMO']
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display original image
        with col1:
            st.subheader("Uploaded Image")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption='Original Image', use_column_width=True)
        
        try:
            # Preprocess image
            img_resized = cv2.resize(img, IMG_SIZE)
            preprocessor = Preprocessor()
            img_processed = preprocessor.preprocess_image(img_resized)
            img_batch = np.expand_dims(img_processed, axis=0)
            
            # Load model
            model_path = get_model_path()
            if model_path is None:
                st.error("Model file not found! Please ensure the model is properly saved.")
                return
                
            model = tf.keras.models.load_model(model_path)
            
            # Make prediction
            with st.spinner('Processing...'):
                predictions = model.predict(img_batch, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx]
                predicted_class = class_names[predicted_class_idx]
            
            # Display results
            with col2:
                st.subheader("Prediction Results")
                st.markdown(f"""
                **Predicted Class:** {predicted_class}  
                **Confidence:** {confidence:.2%}
                """)
                
                # Show top 3 predictions
                st.subheader("Top 3 Predictions")
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                for idx in top_3_idx:
                    st.markdown(f"**{class_names[idx]}:** {predictions[0][idx]:.2%}")
                
                # Create and display confidence chart
                fig = create_confidence_chart(predictions[0], class_names)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate and display clinical report
                st.markdown("---")
                report = create_report(
                    st.session_state.patient_data,
                    predicted_class,
                    confidence,
                    top_3_idx,
                    class_names
                )
                st.markdown(report)
                
                # Add download button for report
                report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=report_filename,
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    
    # Add explanation of cell types
    with st.expander("Cell Type Descriptions"):
        st.markdown("""
        ### Bone Marrow Cell Types
        
        | Cell Code | Cell Type | Description |
        |:----------|:----------|:------------|
        | ABE | Abnormal Eosinophil | Eosinophil with abnormal morphology |
        | ART | Artifact | Non-cellular debris or processing artifact |
        | BAS | Basophil | Granulocyte with basophilic granules |
        | BLA | Blast | Immature hematopoietic cell, potential leukemia indicator |
        | EBO | Erythroblast | Nucleated red blood cell precursor |
        | EOS | Eosinophil | Granulocyte with eosinophilic granules |
        | FGC | Faggot Cell | Cell with Auer rod bundles, pathognomonic for APL |
        | HAC | Hairy Cell | Characteristic of hairy cell leukemia |
        | KSC | Smudge Cell | Damaged lymphocyte with disrupted membrane |
        | LYI | Lymphocyte | Mature lymphocyte |
        | LYT | Lymphoblast | Immature lymphocyte, potential ALL indicator |
        | MMZ | Metamyelocyte | Granulocyte precursor |
        | MON | Monocyte | Mature monocyte |
        | MYB | Myeloblast | Myeloid precursor, potential AML indicator |
        | NGB | Band Neutrophil | Immature neutrophil with band-shaped nucleus |
        | NGS | Segmented Neutrophil | Mature neutrophil with segmented nucleus |
        | NIF | Not In Focus | Image quality issue |
        | OTH | Other | Cell type not in standard classification |
        | PEB | Proerythroblast | Early erythroid precursor |
        | PLM | Plasma Cell | Antibody-producing cell, elevated in myeloma |
        | PMO | Promyelocyte | Early myeloid precursor |
        """)

if __name__ == "__main__":
    main()