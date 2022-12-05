from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
import pandas as pd
from dscharts import get_variable_types
import category_encoders as ce
from pandas import concat, DataFrame
from sklearn.preprocessing import OneHotEncoder
from numpy import number

file = ""

def data_encoding(filename, na_values):
    file = filename
    register_matplotlib_converters()
    data = read_csv(filename, na_values=na_values, parse_dates=True, infer_datetime_format=True)
    #get_unique_values(data)
    variable_encoding(data)
    

def get_unique_values(data):
    symbolic_vars = get_variable_types(data)['Symbolic']

    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    for n in range(len(symbolic_vars)):
        dataByColumn = data[symbolic_vars[n]]
        unique = pd.unique(dataByColumn) 
        print(symbolic_vars[n], unique)

def variable_encoding(data):
    #Race enconded
    if 'race' in data:
        encoder= ce.OrdinalEncoder(cols=['race'],return_df=True,mapping=[{'col':'race',
        'mapping':{'Caucasian':0,'AfricanAmerican':1,'Other':2,'Asian':3, 'Hispanic':4}}])
        data['race'] = encoder.fit_transform(data["race"])

    #Gender encoded
    if 'gender' in data:
        encoder= ce.OrdinalEncoder(cols=['gender'],return_df=True, mapping=[{'col':'gender',
        'mapping':{'Female':0,'Male':1 }}])
        data['gender'] = encoder.fit_transform(data["gender"])

    #Age encoded
    if 'age' in data:
        encoder= ce.OrdinalEncoder(cols=['age'],return_df=True, mapping=[{'col':'age',
        'mapping':{'[0-10)':0,'[10-20)':1, '[20-30)':2,'[30-40)':3, '[40-50)':4,'[50-60)':5, '[60-70)':6,'[70-80)':7,
                '[80-90)': 8, '[90-100)': 9}}])
        data['age'] = encoder.fit_transform(data["age"])

    #Weight encoded
    if 'weight' in data:
        encoder= ce.OrdinalEncoder(cols=['weight'],return_df=True, mapping=[{'col':'weight',
        'mapping':{'[0-25)':0,'[25-50)':1, '[50-75)':2,'[75-100)':3, '[100-125)':4,'[125-150)':5, '[150-175)':6,'[175-200)':7,
                '>200': 8}}])
        data['weight'] = encoder.fit_transform(data["weight"])   

    #payer_code encoded
    if 'payer_code' in data:
        encoder= ce.OrdinalEncoder(cols=['payer_code'],return_df=True, mapping=[{'col':'payer_code',
        'mapping':{'MC':0,'MD':1, 'HM':2,'UN':3, 'BC':4,'SP':5, 'CP':6,'SI':7, 'DM': 8, 'CM': 9, 'CH': 10, 'PO': 11,
            'WC': 12, 'OT': 13, 'OG': 14, 'MP': 15, 'FR': 16}}])
        data['payer_code'] = encoder.fit_transform(data["payer_code"]) 

    #medical_specialty encoded
    if 'medical_specialty' in data:
        encoder= ce.OrdinalEncoder(cols=['medical_specialty'],return_df=True, mapping=[{'col':'medical_specialty',
            'mapping':{'Pediatrics-Endocrinology': 0, 'InternalMedicine': 1,
            'Family/GeneralPractice': 2, 'Cardiology': 3, 'Surgery-General' : 4, 'Orthopedics': 5,
            'Gastroenterology': 6, 'Surgery-Cardiovascular/Thoracic': 7, 'Nephrology': 8,
            'Orthopedics-Reconstructive': 9, 'Psychiatry': 10, 'Emergency/Trauma': 11,
            'Pulmonology': 12, 'Surgery-Neuro': 13, 'Obsterics&Gynecology-GynecologicOnco' : 14,
            'ObstetricsandGynecology': 15, 'Pediatrics': 16, 'Hematology/Oncology': 17,
            'Otolaryngology': 18, 'Surgery-Colon&Rectal': 19, 'Pediatrics-CriticalCare': 20,
            'Endocrinology': 21, 'Urology': 22, 'Psychiatry-Child/Adolescent': 23,
            'Pediatrics-Pulmonology': 24, 'Neurology': 25, 'Anesthesiology-Pediatric': 26,
            'Radiology': 27, 'Pediatrics-Hematology-Oncology': 28, 'Psychology': 29, 'Podiatry': 30,
            'Gynecology': 31, 'Oncology': 32, 'Pediatrics-Neurology': 33, 'Surgery-Plastic': 34,
            'Surgery-Thoracic': 35, 'Surgery-PlasticwithinHeadandNeck': 36, 'Ophthalmology': 37,
            'Surgery-Pediatric': 38, 'Pediatrics-EmergencyMedicine': 39,
            'PhysicalMedicineandRehabilitation': 40, 'InfectiousDiseases': 41, 'Anesthesiology': 42,
            'Rheumatology': 43, 'AllergyandImmunology': 44, 'Surgery-Maxillofacial': 45,
            'Pediatrics-InfectiousDiseases': 46, 'Pediatrics-AllergyandImmunology': 47,
            'Dentistry': 48, 'Surgeon': 49, 'Surgery-Vascular': 50, 'Osteopath': 51,
            'Psychiatry-Addictive': 52, 'Surgery-Cardiovascular': 53, 'PhysicianNotFound': 54,
            'Hematology': 55, 'Proctology': 56, 'Obstetrics': 57, 'SurgicalSpecialty': 58, 'Radiologist': 59,
            'Pathology': 60, 'Dermatology': 61, 'SportsMedicine': 62, 'Speech': 63, 'Hospitalist': 64,
            'OutreachServices': 65, 'Cardiology-Pediatric': 66, 'Perinatology': 67,
            'Neurophysiology': 68, 'Endocrinology-Metabolism': 69, 'DCPTEAM': 70, 'Resident': 71}}])
        data['medical_specialty'] = encoder.fit_transform(data["medical_specialty"])

    #max_glu_serum encoded
    if 'max_glu_serum' in data:
        encoder= ce.OrdinalEncoder(cols=['max_glu_serum'],return_df=True, mapping=[{'col':'max_glu_serum',
        'mapping':{'>200':0,'Norm':1, '>300':2 }}])
        data['max_glu_serum'] = encoder.fit_transform(data["max_glu_serum"])

    #A1Cresult encoded
    if 'A1Cresult' in data:
        encoder= ce.OrdinalEncoder(cols=['A1Cresult'],return_df=True, mapping=[{'col':'A1Cresult',
        'mapping':{'>7':0,'Norm':1, '>8':2 }}])
        data['A1Cresult'] = encoder.fit_transform(data["A1Cresult"])

    #Variable encoding for Medication
    medication = {'No':0,'Down':1, 'Steady':2, 'Up': 3 }

    if 'metformin' in data:
        encoder= ce.OrdinalEncoder(cols=['metformin'],return_df=True, mapping=[{'col':'metformin','mapping': medication}])
        data['metformin'] = encoder.fit_transform(data["metformin"])

    if 'repaglinide' in data:
        encoder= ce.OrdinalEncoder(cols=['repaglinide'],return_df=True, mapping=[{'col':'repaglinide','mapping': medication}])
        data['repaglinide'] = encoder.fit_transform(data["repaglinide"])

    if 'nateglinide' in data:
        encoder= ce.OrdinalEncoder(cols=['nateglinide'],return_df=True, mapping=[{'col':'nateglinide','mapping': medication}])
        data['nateglinide'] = encoder.fit_transform(data["nateglinide"])

    if 'chlorpropamide' in data:
        encoder= ce.OrdinalEncoder(cols=['chlorpropamide'],return_df=True, mapping=[{'col':'chlorpropamide','mapping': medication}])
        data['chlorpropamide'] = encoder.fit_transform(data["chlorpropamide"])

    if 'glimepiride' in data:
        encoder= ce.OrdinalEncoder(cols=['glimepiride'],return_df=True, mapping=[{'col':'glimepiride','mapping': medication}])
        data['glimepiride'] = encoder.fit_transform(data["glimepiride"])

    if 'glipizide' in data:
        encoder= ce.OrdinalEncoder(cols=['glipizide'],return_df=True, mapping=[{'col':'glipizide','mapping': medication}])
        data['glipizide'] = encoder.fit_transform(data["glipizide"])

    if 'glyburide' in data:
        encoder= ce.OrdinalEncoder(cols=['glyburide'],return_df=True, mapping=[{'col':'glyburide','mapping': medication}])
        data['glyburide'] = encoder.fit_transform(data["glyburide"])

    if 'pioglitazone' in data:
        encoder= ce.OrdinalEncoder(cols=['pioglitazone'],return_df=True, mapping=[{'col':'pioglitazone','mapping': medication}])
        data['pioglitazone'] = encoder.fit_transform(data["pioglitazone"])

    if 'rosiglitazone' in data:
        encoder= ce.OrdinalEncoder(cols=['rosiglitazone'],return_df=True, mapping=[{'col':'rosiglitazone','mapping': medication}])
        data['rosiglitazone'] = encoder.fit_transform(data["rosiglitazone"])
    
    if 'acarbose' in data:
        encoder= ce.OrdinalEncoder(cols=['acarbose'],return_df=True, mapping=[{'col':'acarbose','mapping': medication}])
        data['acarbose'] = encoder.fit_transform(data["acarbose"])

    if 'miglitol' in data:
        encoder= ce.OrdinalEncoder(cols=['miglitol'],return_df=True, mapping=[{'col':'miglitol','mapping': medication}])
        data['miglitol'] = encoder.fit_transform(data["miglitol"])

    if 'tolazamide' in data:
        encoder= ce.OrdinalEncoder(cols=['tolazamide'],return_df=True, mapping=[{'col':'tolazamide','mapping': medication}])
        data['tolazamide'] = encoder.fit_transform(data["tolazamide"])

    if 'examide' in data:
        encoder= ce.OrdinalEncoder(cols=['examide'],return_df=True, mapping=[{'col':'examide','mapping': medication}])
        data['examide'] = encoder.fit_transform(data["examide"])

    if 'citoglipton' in data:
        encoder = ce.OrdinalEncoder(cols=['citoglipton'],return_df=True, mapping=[{'col':'citoglipton','mapping': medication}])
        data['citoglipton'] = encoder.fit_transform(data["citoglipton"])

    if 'insulin' in data:
        encoder= ce.OrdinalEncoder(cols=['insulin'],return_df=True, mapping=[{'col':'insulin','mapping': medication}])
        data['insulin'] = encoder.fit_transform(data["insulin"])

    if 'glyburide-metformin' in data:
        encoder= ce.OrdinalEncoder(cols=['glyburide-metformin'],return_df=True, mapping=[{'col':'glyburide-metformin','mapping': medication}])
        data['glyburide-metformin'] = encoder.fit_transform(data["glyburide-metformin"])

    if 'chlorpropamide' in data:    
        encoder= ce.OrdinalEncoder(cols=['chlorpropamide'],return_df=True, mapping=[{'col':'chlorpropamide','mapping': medication}])
        data['chlorpropamide'] = encoder.fit_transform(data["chlorpropamide"])

    if 'tolazamide' in data:    
        encoder= ce.OrdinalEncoder(cols=['tolazamide'],return_df=True, mapping=[{'col':'tolazamide','mapping': medication}])
        data['tolazamide'] = encoder.fit_transform(data["tolazamide"])

    if 'glipizide-metformin' in data:
        encoder= ce.OrdinalEncoder(cols=['glipizide-metformin'],return_df=True, mapping=[{'col':'glipizide-metformin','mapping': medication}])
        data['glipizide-metformin'] = encoder.fit_transform(data["glipizide-metformin"])

    if 'metformin-pioglitazone' in data:
        encoder= ce.OrdinalEncoder(cols=['metformin-pioglitazone'],return_df=True, mapping=[{'col':'metformin-pioglitazone','mapping': medication}])
        data['metformin-pioglitazone'] = encoder.fit_transform(data["metformin-pioglitazone"])

    change = {'No':0,'Ch':1 }
    if 'change' in data:
        encoder= ce.OrdinalEncoder(cols=['change'],return_df=True, mapping=[{'col':'change','mapping': change}])
        data['change'] = encoder.fit_transform(data["change"])

    diabetesMed = {'No':0,'Yes':1 }
    if 'diabetesMed' in data:
        encoder= ce.OrdinalEncoder(cols=['diabetesMed'],return_df=True, mapping=[{'col':'diabetesMed','mapping': diabetesMed}])
        data['diabetesMed'] = encoder.fit_transform(data["diabetesMed"])

    if 'acetohexamide' in data:
        data = data.drop("acetohexamide", axis='columns')
    if 'tolbutamide' in data:
        data = data.drop("tolbutamide", axis='columns')
    if 'troglitazone' in data:    
        data = data.drop("troglitazone", axis='columns')
    if 'glimepiride-pioglitazone' in data:
        data = data.drop("glimepiride-pioglitazone", axis='columns')
    if 'metformin-rosiglitazone' in data:
        data = data.drop("metformin-rosiglitazone", axis='columns')

    data["diag_1"] = diagnostic(data['diag_1']).values()
    data["diag_2"] = diagnostic(data['diag_2']).values()
    data["diag_3"] = diagnostic(data['diag_3']).values()

    variables = get_variable_types(data)
    symbolic_vars = variables['Symbolic']
    #df = dummify(data, symbolic_vars)
    data["readmitted"] = readmitted(data['readmitted']).values()
    data.to_csv('data/classification/lab2_datasets/dataset1/diabetic_data_variable_enconding.csv', index=False)

#Codifying Diag_1, Diag_2, Diag_3 Variables, using ICD9 Codification
#http://www.icd9data.com/2015/Volume1/default.htm
def diagnostic(data_Diagnostic) -> dict:
    data_diag_encoding = {}
    for n in range(len(data_Diagnostic)):
        #print(data_Diagnostic[n])

        if pd.isna(data_Diagnostic[n]):
            data_diag_encoding[n] = -1
        elif "V" in data_Diagnostic[n]:
            data_diag_encoding[n] = 17
        elif "E" in data_Diagnostic[n]:
            data_diag_encoding[n] = 18
        else:
        #    print(type(data_Diagnostic[n]))
            value = float(data_Diagnostic[n])
        #    print(value)
            if value >= 0 and value <= 139:
                data_diag_encoding[n] = 0
            elif value >= 140 and value <= 239:
                data_diag_encoding[n] = 1
            elif value >= 240 and value <= 279:
                data_diag_encoding[n] = 2
            elif value >= 280 and value <= 289:
                data_diag_encoding[n] = 3
            elif value >= 290 and value <= 319:
                data_diag_encoding[n] = 4
            elif value >= 320 and value <= 389:
                data_diag_encoding[n] = 5
            elif value >= 390 and value <= 459:
                data_diag_encoding[n] = 6
            elif value >= 460 and value <= 519:
                data_diag_encoding[n] = 7
            elif value >= 520 and value <= 579:
                data_diag_encoding[n] = 8
            elif value >= 580 and value <= 629:
                data_diag_encoding[n] = 9
            elif value >= 630 and value <= 679:
                data_diag_encoding[n] = 10
            elif value >= 680 and value <= 709:
                data_diag_encoding[n] = 11
            elif value >= 710 and value <= 739:
                data_diag_encoding[n] = 12
            elif value >= 740 and value <= 759:
                data_diag_encoding[n] = 13
            elif value >= 760 and value <= 779:
                data_diag_encoding[n] = 14
            elif value >= 780 and value <= 799:
                data_diag_encoding[n] = 15
            elif value >= 800 and value <= 999:
                data_diag_encoding[n] = 16       

    return data_diag_encoding

def readmitted(data_readmitted) -> dict:
    data_readmitted_encoding = {}
    for n in range(len(data_readmitted)):
        if pd.isna(data_readmitted[n]):
            data_readmitted_encoding[n] = -1
        elif "NO" == data_readmitted[n]:
            data_readmitted_encoding[n] = 0
        else:
            data_readmitted_encoding[n] = 1
    return data_readmitted_encoding

def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

#data_encoding('data/classification/lab2_datasets/dataset1/drop_recs_cols_dataset1.csv', "?")
#data_encoding('data/classification/lab2_datasets/dataset1/mv_filled_most_frequent_dataset1.csv', "?")
data_encoding('data/diabetic_data_drop_outliers.csv', "?")
