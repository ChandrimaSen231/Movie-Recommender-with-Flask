import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from flask import Flask,render_template,request

def get_movie_list():
    movie_title = pd.read_csv("ml-100k/u.item",sep="|",header=None
                     ,encoding='latin',names=["movie id","movie title"],
                     usecols=[0,1],index_col=0)

    return movie_title

def get_data():

    ratings = pd.read_csv("ml-100k/u.data",sep="\t",header=None
                     ,names=['user','movie_id','rating'],usecols=[0,1,2],index_col = 1)

    movie_ratings = ratings.groupby("movie_id").agg({'user':'count','rating':'mean'})
    movie_ratings.rename({'user':'count','rating':'avg_rating'},axis = 1,inplace=True)

    mi = movie_ratings['count'].min()
    mx = movie_ratings['count'].max()

    movie_ratings['popularity'] = movie_ratings['count'].apply(lambda v:round((v - mi)/(mx - mi),2))

    cols = " movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western"
    cols = cols.split(' | ')

    movies = pd.read_csv("ml-100k/u.item",sep="|",header=None
                     ,encoding='latin',names=cols,index_col=0)
    movies.drop(['movie title','release date','video release date','IMDb URL'],axis= 1,inplace = True)

    movie_title = get_movie_list()
    movie_ratings.drop(['count'],axis=1,inplace=True)
    X = pd.merge(movies,movie_ratings,left_index=True,right_index=True)
    return X,movie_title

def similarmovies(mname,k=5):
    X,movie_title = get_data()
    mid = movie_title[movie_title['movie title'] == mname].index[0]
    X_new = X.loc[movie_title['movie title'] == mname].values[0]
    result = []
    for ix,movie,pop in zip(X.index,X.values,X['popularity'].to_numpy()):
        if ix != mid:
            d = euclidean(X_new,movie)
            result.append((d,pop,ix))
        
    result.sort(key = lambda x: (x[0],x[1]))
    b = [i for d,p,i in result[:k]]
    
    movie_title_pop = pd.concat([movie_title.loc[b],X['popularity'].loc[b]],axis=1)
    return movie_title_pop


app = Flask(__name__)
@app.route('/')
def home():
    movie = get_movie_list()
    m = [{'movie' : 'Select Movie'}]
    for i in range(len(movie["movie title"])):
        m.append({'movie' : movie['movie title'].iloc[i]})
    return render_template('recomm.html',data= m )

@app.route('/show_recom',methods=["POST"])
def show_recom():
    name,val = [x for x in request.form.values()]
    movies = similarmovies(name,int(val))
    movie = get_movie_list()
    m = [{'movie' : 'Select Movie'}]
    for i in range(len(movie["movie title"])):
        m.append({'movie' : movie['movie title'].iloc[i]})
    text = []
    for i in range(int(val)):
        text.append(f"{i+1}. "+movies['movie title'].iloc[i])
    return render_template('recomm.html',movie_text = text,data =m)

if __name__ == '__main__':
    app.run(debug=True)