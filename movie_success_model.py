import pandas as pd


#load dataset
movies = pd.read_csv("Data/tmdb_5000_movies.csv")
credits = pd.read_csv("Data/tmdb_5000_credits.csv")
#print(movies.head()) #view first rows
#print(credits.head())

#Merge movies and credits on movie_id 
merged_df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")

#Drop duplicate 'movie_id' column
merged_df.drop(columns=['movie_id'], inplace=True)

#print(merged_df.head())

#create a 'success' column: 1 = Hit, 0 = Flop
#Hit : revenue >= 2x Budget

merged_df['success'] = (merged_df['revenue'] >= 2* merged_df['budget']).astype(int)

# Rename 'title_x' to 'title' and drop 'title_y'
merged_df.rename(columns={'title_x': 'title'}, inplace=True)
merged_df.drop(columns=['title_y'], inplace=True)

#print(merged_df[['title', 'budget', 'revenue', 'success']].head())

#EDA
#Missing values
missing_values = merged_df.isnull().sum()

# Display only columns with missing values
print(missing_values[missing_values > 0])
#Drop unnecessary columns
merged_df.drop(columns=['homepage', 'tagline'], inplace=True)
#Fill missing 'overview' with an empty string 
merged_df['overview'].fillna('', inplace=True)
#Drop rows where 'release_date' is missing 
merged_df.dropna(subset=['release_date'], inplace=True)
#Fill missing 'runtime' with the median value
merged_df['runtime'] = merged_df['runtime'].fillna(merged_df['runtime'].median())
#check if missing values are handled

#print(merged_df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
#Plot distributions for budget, revenue, and runtime
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Budget distribution
sns.histplot(merged_df['budget'], bins=50, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Budget Distribution')
axes[0].set_xlabel('Budget ($)')

# Revenue distribution
sns.histplot(merged_df['revenue'], bins=50, kde=True, ax=axes[1], color='green')
axes[1].set_title('Revenue Distribution')
axes[1].set_xlabel('Revenue ($)')

# Runtime distribution
sns.histplot(merged_df['runtime'], bins=30, kde=True, ax=axes[2], color='red')
axes[2].set_title('Runtime Distribution')
axes[2].set_xlabel('Runtime (minutes)')

#plt.show()

#compute correlation matrix 
correlation_matrix = merged_df[['budget', 'revenue', 'runtime', 'success']].corr()

#plot correlation heatmap 
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature correlation Heatmap")
plt.show()


#Feature Engineering 
#Profit Margin : (Revenue - Budget)/Budget 
merged_df['profit_margin'] = (merged_df['revenue'] - merged_df['budget']) / (merged_df['budget'] + 1)
#Extract Release Month 
merged_df['release_date'] = pd.to_datetime(merged_df['release_date'], errors='coerce')
merged_df['release_month'] = merged_df['release_date'].dt.month

#One-Hot Encode Genres 
import ast 

#convert 'genres' from string to list
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genres]
    except (ValueError, SyntaxError): 
        return [] #if it's invalid return an empty list

merged_df['genres_list'] = merged_df['genres'].apply(extract_genres)

#one-hot encoding genres
genres_dummies = merged_df['genres_list'].explode().str.get_dummies().groupby(level=0).sum()

#Merge with the main dataframe
merged_df = pd.concat([merged_df, genres_dummies], axis=1)

#Drop the original genres column 
merged_df.drop(columns = ['genres', 'genres_list'], inplace=True)

#check new features
print(merged_df[['profit_margin', 'release_month'] + list(genres_dummies.columns)].head())

#Training
from sklearn.model_selection import train_test_split

#Select relevant features for training 
features = ['budget', 'runtime', 'profit_margin', 'release_month'] + list(genres_dummies.columns)
target = 'success' 

#split dataset
X_train , X_test , y_train, y_test = train_test_split(merged_df[features], merged_df[target], test_size=0.2, random_state=42, stratify=merged_df[target])

#check shapes
print(X_train.shape , X_test.shape, y_train.shape , y_test.shape)


#Train randomForest model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predections
y_pred = model.predict(X_test)

#Evaluate model 
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"New Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("New Classification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "movies_success_model.pkl")

print(model.feature_names_in_)

