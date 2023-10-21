import os
import json
import csv
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required,\
                        logout_user,current_user 
from flask_sqlalchemy import SQLAlchemy

from werkzeug.urls import url_encode 
from werkzeug.security import generate_password_hash, check_password_hash




file_path = os.path.abspath(os.getcwd())+"\database.db"

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+file_path
db = SQLAlchemy(app)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(64), unique=True)
    password_hash = db.Column(db.String(128))
    recommendations = db.relationship('Recommendation', backref='user', lazy='dynamic')

    def set_password(self,password):
        self.password_hash=generate_password_hash(password)

    def check_password(self,password):
        return check_password_hash(self.password_hash,password)
		
		
class Recommendation(db.Model):
    id=db.Column(db.Integer(),primary_key=True)
    book_title=db.Column(db.String(128))
    author=db.Column(db.String(128))
    user_id=db.Column(db.Integer(),db.ForeignKey('user.id'))		
		

data = []
with open("booksummaries.txt", "r", encoding='utf-8') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in reader:
        data.append(row)

# convert data to pandas dataframe
books = pd.DataFrame.from_records(data, columns=['book_id', 'freebase_id', 'book_title', 'author', 'publication_date', 'genre', 'summary'])
books.head()
def parse_genre_entry(genre_info):
    if genre_info == '':
        return []
    genre_dict = json.loads(genre_info)
    genres = list(genre_dict.values())
    return genres

books['genre'] = books['genre'].apply(parse_genre_entry)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
books['summary'] = books['summary'].fillna('')

tfidf_matrix = tfidf.fit_transform(books['summary'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# sklearn TF-IDF defaults to using L2 Norm, for which linear kernel == cosine similarity

def clean_flatten(data):
    cleaned = []
    for entry in data:
        # strip spaces and flatten into small caps
        cleaned.append(str.lower(entry.replace(' ', '')))
    return cleaned

books['genre_kws'] = books['genre'].apply(clean_flatten)
books['author_kws'] = books['author'].apply(lambda x: str.lower(x.replace(' ', '')))

def merge_kws(df):
    return ' '.join(df.genre_kws) + ' ' + df.author_kws

books['kws'] = books.apply(merge_kws, axis=1)

indices = pd.Series(books.index, index=books['book_title']).drop_duplicates()


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(books['kws'])
kw_similarity = cosine_similarity(count_matrix, count_matrix)

def recommend(titles, similarity_matrix, topk=10):
    book_indices = [indices[title] for title in titles]
    
    # Initialize combined_vector as an array of zeros with appropriate length
    combined_vector = np.zeros(similarity_matrix.shape[1])
    
    # Add up the vectors for each title
    for index in book_indices:
        sim_vector = similarity_matrix[index]
        print(f"Shape of sim_vector: {sim_vector.shape}")  # Print shape for debugging
        
        if sim_vector.shape != combined_vector.shape:
            print(f"Skipping index {index} due to shape mismatch")
            continue  # Skip this iteration if shapes do not match
        
        combined_vector += sim_vector
        
    combined_vector /= len(book_indices)  # Normalize by the number of titles

    similarity_scores = list(enumerate(combined_vector))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_scores = similarity_scores[:topk]  # Get top k scores
    
    recommendation_indices = [i[0] for i in top_scores]

    return books['book_title'].iloc[recommendation_indices]

titles_to_recommend = ['The Stranger']
recs = recommend(titles_to_recommend, kw_similarity)
# print(recs)
books.iloc[recs.index]


# Your book recommendation code here

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_book_titles')
def get_book_titles():
    # Get the current input from the request parameters.
    current_input = request.args.get('term', '')

    # Get all unique book titles from your dataframe that contain the current input.
    matching_titles = books[books['book_title'].str.contains(current_input, case=False)]['book_title'].unique().tolist()

    return jsonify(matching_titles)

# Define the route for the recommendation results page
# Rename the Flask route handler to avoid conflict
@app.route('/recommend_books', methods=['POST'])
@login_required
def recommend_books():
    # Get the book titles as a comma-separated string from the input form
    titles_input = request.form.get('book_titles')

    # Split the input string into a list of book titles
    titles_to_recommend = [title.strip() for title in titles_input.split(',')]

    recommended_books = {}  # Create a dictionary to store recommendations

    for title in titles_to_recommend:
        recs = recommend([title], kw_similarity)
        recommended_books[title] = books.iloc[recs.index]
        for _, row in books.iloc[recs.index].iterrows():
            new_rec=Recommendation(book_title=row['book_title'], author=row['author'], user_id=current_user.id)
            db.session.add(new_rec) 
            db.session.commit() 

    return render_template('recommendation.html', books=recommended_books)

@login_manager.user_loader
def load_user(user_id):
   return User.query.get(int(user_id))


@app.route('/previous_recommendations')
@login_required
def previous_recommendations():
    # Get all recommendations for the current user.
    all_recommendations = Recommendation.query.filter_by(user_id=current_user.id).all()

    # Create a dictionary where keys are book titles and values are authors.
    # This will automatically remove duplicates because dictionaries cannot have duplicate keys.
    unique_recommendations = {rec.book_title: rec.author for rec in all_recommendations}

    return render_template('previous_recommendations.html', recommendations=unique_recommendations.items())



@app.route('/register', methods=['GET','POST'])
def register():
   if request.method=='POST':
       username=request.form.get('username')
       password=request.form.get('password')
       if not username or not password:
           flash("Username or Password cannot be empty.")
           return redirect(url_for('register'))
       existing_user=User.query.filter_by(username=username).first()
       if existing_user is None:
          new_user=User(username=username)
          new_user.set_password(password) 
          db.session.add(new_user) 
          db.session.commit() 
          return redirect(url_for('login'))  
   return render_template("register.html")

@app.route('/login',methods=['GET','POST'])
def login():
   if request.method=='POST':
      username=request.form.get('username')
      password=request.form.get('password')
      remember=bool(request.form.get("remember", False))
      user=User.query.filter_by(username=username).first()
      if not user or not user.check_password(password):
         flash("Invalid username or password.")
         return redirect(url_for('login'))
      login_user(user, remember=remember)
      return redirect(url_for('home')) 
   return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


if __name__ == '__main__':
	with app.app_context():
		db.create_all()
	app.run(debug=True)



