import pandas as pd
import shutil

#create a copy of the original database
shutil.copy("data/survey.csv","data/survey_cpy.csv")
#load the copy
df = pd.read_csv("data/survey_cpy.csv")

#clean the unformatted values from GENDER column
def clean_gender(val):
    val=str(val).strip().lower()
    if val in ['m','male','maile','cis male','mal','male (cis)','make','man','msle','mail','malr','cis man']:
        return "Male"
    if val in ['female','cis female','f','woman','femake','cis female/femme','female (cis)','femail']:
        return "Female"
    else:
        return "Other"
df['Gender']=df['Gender'].apply(clean_gender)

#clean the unformatted values from AGE column
df=df[(df['Age']>=18)&(df['Age']<=100)]

#clean the ones not being in a tech company
df=df[(df['tech_company']=='Yes')]

#fill missing self_employed, work_interfere values with "Unknown"
df['self_employed']=df['self_employed'].fillna('Unknown')
df['work_interfere']=df['work_interfere'].fillna('Unknown')

print(f"\nThere are {len(df)} rows left\n")
df.to_csv("data/cleaned_survey.csv", index=False)