import pandas as pd

people = pd.read_csv('01_people.csv')
education = pd.read_csv('03_education.csv')
experience = pd.read_csv('04_experience.csv')
person_skills = pd.read_csv('05_person_skills.csv')
skills = pd.read_csv('06_skills.csv')

skills_merged = person_skills.merge(skills, on='skill', how='left')

skills_grouped = skills_merged.groupby('person_id')['skill'].apply(lambda x: ', '.join(x.dropna().unique())).reset_index()

df = people \
    .merge(education, on='person_id', how='left') \
    .merge(experience, on='person_id', how='left') \
    .merge(skills_grouped, on='person_id', how='left')

df.fillna('', inplace=True)


columns_to_keep = ['name', 'experience_details', 'education_details', 'skill']
final_df = df[[col for col in columns_to_keep if col in df.columns]]

final_df.to_csv('final_cleaned.csv', index=False)

print(" Final cleaned file created with only name, education, experience, and skill columns.")
