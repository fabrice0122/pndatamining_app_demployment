from flask import Flask,render_template,request 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel    
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel      
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('my_model.h5')
app = Flask(__name__)
def load_data(data):
    df= pd.read_csv(data, sep= ';', error_bad_lines= False, encoding= 'latin-1')
    df=df.head(500)
    return df

def search_term_if_not_found(term,df):
    term = term.capitalize()
    result_df= df[df['Book-Title'].str.contains(term)]
    return result_df['Book-Title'].iloc[0]    

def vectorize_text_to_cosine_max(data):
    count_vec= CountVectorizer()
    cv_mat= count_vec.fit_transform(data)
    cosine_sim=cosine_similarity(cv_mat)
    return cosine_sim
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=8):
    course_indices=pd.Series(df.index,index=df['Book-Title']).drop_duplicates()
    idx=course_indices[title]
    sim_scores=list(enumerate(cosine_sim_mat[idx]))
    sim_scores= sorted(sim_scores,key=lambda x:x[1],reverse=True)
    selected_course_indices=[i[0] for i in sim_scores[1:]]
    selected_course_score=[i[0] for i in sim_scores[1:]]
    result_df= df.iloc[selected_course_indices]   
    result_df['similarity score']=selected_course_score
    final_recommeded= result_df[['Book-Title','Book-Author','Year-Of-Publication','similarity score','Image-URL-L']]
    return final_recommeded.head(num_of_rec)
df=load_data('https://raw.githubusercontent.com/tttgm/fellowshipai/master/book_crossing_dataset/BX-Books.csv')
cosine_sim_mat=vectorize_text_to_cosine_max(df['Book-Title'])
def get_suggestions():
    data = pd.read_csv("https://raw.githubusercontent.com/sahilpocker/Book-Recommender-System/master/Dataset/books.csv")
    return list(data['title'].str.capitalize())
@app.route('/')
def login():
    return render_template("login.html")
database={'diane':'123','james':'aac','karthik':'asdsf'}

@app.route('/form_login',methods=['POST','GET'])
def login_page():
  df=pd.read_csv('https://raw.githubusercontent.com/Diane10/movies/main/most_rated_books_summary_noerros.csv')
  titles = df['book_title']
  authors=df['book_author']
  years=df['year_of_publication']
  # scores=df['similarity score']
  images = df['image_url_l']
  df_rating=pd.read_csv('https://raw.githubusercontent.com/Diane10/movies/main/books_summary_noerros.csv')
  titles_rating = df_rating['book_title']
  authors_rating=df_rating['book_author']
  years_rating=df_rating['year_of_publication']
  images_rating = df_rating['image_url_l']
  name1=request.form['username']
  sytem=request.form['sytem']
  pwd=request.form['password']
  # COLLABORATIVE
  df= pd.read_csv('https://raw.githubusercontent.com/Diane10/movies/main/Finalcollab.csv')
  coll_titles = df['book_title']
  coll_authors=df['book_author']
  coll_years=df['year_of_publication']
  coll_images= df['image_url_l']

  if name1 not in database:
    return render_template('login.html',info='Invalid User')
  else:
    if database[name1]!=pwd:
      return render_template('login.html',info='Invalid Password')
    else:
      if sytem =="content based":
        return render_template('content.html',coll_images=coll_images,coll_years=coll_years,coll_titles=coll_titles,coll_authors=coll_authors,name=name1,title = titles,author=authors,year = years,image=images,titles_rating=titles_rating,authors_rating=authors_rating,years_rating=years_rating,images_rating=images_rating)
      elif sytem == "collaborative based":
        return render_template('collaborative.html',coll_images=coll_images,coll_years=coll_years,coll_titles=coll_titles,coll_authors=coll_authors,name=name1,title = titles,author=authors,year = years,image=images,titles_rating=titles_rating,authors_rating=authors_rating,years_rating=years_rating,images_rating=images_rating)

@app.route('/predict', methods = ['POST']) # /result route Ratingsreviews
def predict():
  name = request.form['book_name']
  searchdf = df[df['Book-Title']== name]
  searchtitles = searchdf['Book-Title']
  searchauthors= searchdf['Book-Author']
  searchyears= searchdf['Year-Of-Publication']
  # scores=result['similarity score']
  searchimages = searchdf['Image-URL-L']
  df_rating=pd.read_csv('https://raw.githubusercontent.com/Diane10/movies/main/mostrated.csv')
  titles_rating = df_rating['book_title']
  authors_rating=df_rating['book_author']
  # years=df['year_of_publication']
  scores_rating=df_rating['ratings']
  images_rating = df_rating['image_url_l']
  if name is not None:
    try :
      result= get_recommendation(name,cosine_sim_mat,df,8)
      titles = result['Book-Title']
      authors=result['Book-Author']
      years=result['Year-Of-Publication']
      # scores=result['similarity score']
      images = result['Image-URL-L']
      suggestions= get_suggestions()
    except:
      name= search_term_if_not_found(name,df)
      searchdf = df[df['Book-Title']== name]
      searchtitles = searchdf['Book-Title']
      searchauthors= searchdf['Book-Author']
      searchyears= searchdf['Year-Of-Publication']
      # scores=result['similarity score']
      searchimages = searchdf['Image-URL-L']
      result= get_recommendation(name,cosine_sim_mat,df,8)
      titles = result['Book-Title']
      authors=result['Book-Author']
      years=result['Year-Of-Publication']
      # scores=result['similarity score']
      images = result['Image-URL-L']
      suggestions= get_suggestions()
  return render_template('Recommender.html',titles_rating=titles_rating,authors_rating=authors_rating, scores_rating=scores_rating,images_rating=images_rating,title = titles,author=authors,year = years,image=images,suggestions=suggestions,searchtitles=searchtitles,searchauthors=searchauthors,searchyears=searchyears,searchimages=searchimages)
@app.route('/content/<title>', methods=['GET'])
def book_content_recommend(title):
  name = str(title)
  if name is not None:
    books_searched=df[df['Book-Title']==name]
    searched_title = books_searched['Book-Title']
    searched_author= books_searched['Book-Author']
    searched_years= books_searched['Year-Of-Publication']
    searched_images= books_searched['Image-URL-L']
    result= get_recommendation(name,cosine_sim_mat,df,8)
    titles = result['Book-Title']
    authors=result['Book-Author']
    years=result['Year-Of-Publication']
    images = result['Image-URL-L']
    return render_template('content_result.html',searched_years=searched_years,searched_images=searched_images,searched_title=searched_title,searched_author=searched_author,title = titles,author=authors,year = years,images=images)

@app.route('/book/<coll_titles>', methods=['GET'])
def book_collaborative_recommend(coll_titles):
  name = str(coll_titles)
  combine_book_rating_data=pd.read_csv('https://raw.githubusercontent.com/Diane10/movies/main/Finalcollab.csv')
  books_df_s=combine_book_rating_data[combine_book_rating_data['book_title']==name]
  titles_searched = books_df_s['book_title']
  authors_searched=books_df_s['book_author']
  year_searched = books_df_s['year_of_publication']
  images_searched = books_df_s['image_url_l']
  user_id = books_df_s['user']
  user_id=user_id.iloc[0]
  user_r = user_id
  b_id =list(combine_book_rating_data.user.unique())
  book_arr = np.array(b_id) #get all book IDs
  user = np.array([user_r for i in range(len(b_id))])
  pred = model.predict([book_arr, user])
  pred = pred.reshape(-1) #reshape to single dimension
  pred_ids = (-pred).argsort()[0:10]
  top10 = combine_book_rating_data.iloc[pred_ids]
  f=['book_title','book_author','year_of_publication','image_url_l']
  displ=(top10[f])
  c_title = displ['book_title']
  c_authors = displ['book_author']
  c_small_image_url= displ['image_url_l']
  c_years= displ['year_of_publication']
  return render_template('result.html',year_searched=year_searched,c_years=c_years,images_searched=images_searched,authors_searched=authors_searched,titles_searched=titles_searched,c_title=c_title,c_authors=c_authors,c_small_image_url=c_small_image_url)
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
