# Project: TMDB_movies Database

## Table of Contents

<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

`<a id='intro'></a>`

## Introduction

> In this report we will walk through **TMDB movies** using a database contains 10,000+ movie, each movie has a set of attributes such as budget, title, director, revenue and so on.
> Using this database we try to know what is the major attributes that can affect on movie industry and how these attributes correlate.

```python
# Importing our Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

```python
df= pd.read_csv('tmdb-movies.csv')
```

```python
df.shape
```

    (10866, 21)

```python
# Checking either the column's values are readable or not 
df.head(2)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>imdb_id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>cast</th>
      <th>homepage</th>
      <th>director</th>
      <th>tagline</th>
      <th>...</th>
      <th>overview</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135397</td>
      <td>tt0369610</td>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>http://www.jurassicworld.com/</td>
      <td>Colin Trevorrow</td>
      <td>The park is open.</td>
      <td>...</td>
      <td>Twenty-two years after the events of Jurassic ...</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>6/9/15</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76341</td>
      <td>tt1392190</td>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...</td>
      <td>http://www.madmaxmovie.com/</td>
      <td>George Miller</td>
      <td>What a Lovely Day.</td>
      <td>...</td>
      <td>An apocalyptic story set in the furthest reach...</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Village Roadshow Pictures|Kennedy Miller Produ...</td>
      <td>5/13/15</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>3.481613e+08</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>

### Main Questions:

> * What are the most three genre produced?
> * How does movie genre and run time affects on movies rate?
> * What are the most and the lowest genres the dirctors like to work on?
> * How does each genre cost and affect on the revenue?
> * What is the the most produced genre in the last year and 1990?
> * what is the relation between movie time and the budget?

`<a id='wrangling'></a>`

## Data Wrangling

> In this section of the report, we will clean our data, trim it and prepare it for answering our questions.

### Assessing Data:

```python
print(f'Number of columns in our database is: {df.shape[0]}')
print(f'Number of columns in our database is: {df.shape[1]}')
```

    Number of columns in our database is: 10866
    Number of columns in our database is: 21

```python
# Checking either column's data types are matching with the values or not
df.dtypes
```

    id                        int64
    imdb_id                  object
    popularity              float64
    budget                    int64
    revenue                   int64
    original_title           object
    cast                     object
    homepage                 object
    director                 object
    tagline                  object
    keywords                 object
    overview                 object
    runtime                   int64
    genres                   object
    production_companies     object
    release_date             object
    vote_count                int64
    vote_average            float64
    release_year              int64
    budget_adj              float64
    revenue_adj             float64
    dtype: object

```python
# Checking the null values
df.isnull().sum()
```

    id                         0
    imdb_id                   10
    popularity                 0
    budget                     0
    revenue                    0
    original_title             0
    cast                      76
    homepage                7930
    director                  44
    tagline                 2824
    keywords                1493
    overview                   4
    runtime                    0
    genres                    23
    production_companies    1030
    release_date               0
    vote_count                 0
    vote_average               0
    release_year               0
    budget_adj                 0
    revenue_adj                0
    dtype: int64

```python
df.nunique()
```

    id                      10865
    imdb_id                 10855
    popularity              10814
    budget                    557
    revenue                  4702
    original_title          10571
    cast                    10719
    homepage                 2896
    director                 5067
    tagline                  7997
    keywords                 8804
    overview                10847
    runtime                   247
    genres                   2039
    production_companies     7445
    release_date             5909
    vote_count               1289
    vote_average               72
    release_year               56
    budget_adj               2614
    revenue_adj              4840
    dtype: int64

```python
# Showing the main statistical attributes for the data
df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>1.086600e+04</td>
      <td>1.086600e+04</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>1.086600e+04</td>
      <td>1.086600e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>66064.177434</td>
      <td>0.646441</td>
      <td>1.462570e+07</td>
      <td>3.982332e+07</td>
      <td>102.070863</td>
      <td>217.389748</td>
      <td>5.974922</td>
      <td>2001.322658</td>
      <td>1.755104e+07</td>
      <td>5.136436e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>92130.136561</td>
      <td>1.000185</td>
      <td>3.091321e+07</td>
      <td>1.170035e+08</td>
      <td>31.381405</td>
      <td>575.619058</td>
      <td>0.935142</td>
      <td>12.812941</td>
      <td>3.430616e+07</td>
      <td>1.446325e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>0.000065</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>1.500000</td>
      <td>1960.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10596.250000</td>
      <td>0.207583</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>90.000000</td>
      <td>17.000000</td>
      <td>5.400000</td>
      <td>1995.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20669.000000</td>
      <td>0.383856</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>99.000000</td>
      <td>38.000000</td>
      <td>6.000000</td>
      <td>2006.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75610.000000</td>
      <td>0.713817</td>
      <td>1.500000e+07</td>
      <td>2.400000e+07</td>
      <td>111.000000</td>
      <td>145.750000</td>
      <td>6.600000</td>
      <td>2011.000000</td>
      <td>2.085325e+07</td>
      <td>3.369710e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>417859.000000</td>
      <td>32.985763</td>
      <td>4.250000e+08</td>
      <td>2.781506e+09</td>
      <td>900.000000</td>
      <td>9767.000000</td>
      <td>9.200000</td>
      <td>2015.000000</td>
      <td>4.250000e+08</td>
      <td>2.827124e+09</td>
    </tr>
  </tbody>
</table>
</div>

### Asssessing Data Conclusions:

> 1. The data is not complicated
> 2. There are many unnecessary data like id, homepage, tagline and release_date
> 3. The budget and revenue also need to be deleted because there is update for this column
> 4. There is Null values need to be dealed with
> 5. Data types are matching with the data values
> 6. The values need a little adjustement

### Cleaning Data:

```python
# Lets start with dropping unnecessary columns
drop = ['id','imdb_id','budget','release_date','homepage','tagline','overview','keywords','revenue']
df = df.drop(drop,axis = 1)
```

```python
#very well, lets check our columns
print(f'Number of columns in our database is: {df.shape[0]}')
print(f'Number of columns in our database is: {df.shape[1]}')
```

    Number of columns in our database is: 10866
    Number of columns in our database is: 12

```python
df.head(1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>original_title</th>
      <th>cast</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
  </tbody>
</table>
</div>

```python
# renaming the columns
df.rename(columns={'original_title':'title'},inplace=True)
df.rename(columns={'budget_adj':'budget'},inplace=True)
df.rename(columns={'revenue_adj':'revenue'},inplace=True)
df.head(1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>title</th>
      <th>cast</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
  </tbody>
</table>
</div>

```python
# making fuction to know the number of nulls in each column
def cols():
    for col in df:
        print(f'cloumn is: {col} ,Null values are: {df[col].isnull().sum()} , dtype is: {df[col].dtypes}')
cols()
```

    cloumn is: popularity ,Null values are: 0 , dtype is: float64
    cloumn is: title ,Null values are: 0 , dtype is: object
    cloumn is: cast ,Null values are: 76 , dtype is: object
    cloumn is: director ,Null values are: 44 , dtype is: object
    cloumn is: runtime ,Null values are: 0 , dtype is: int64
    cloumn is: genres ,Null values are: 23 , dtype is: object
    cloumn is: production_companies ,Null values are: 1030 , dtype is: object
    cloumn is: vote_count ,Null values are: 0 , dtype is: int64
    cloumn is: vote_average ,Null values are: 0 , dtype is: float64
    cloumn is: release_year ,Null values are: 0 , dtype is: int64
    cloumn is: budget ,Null values are: 0 , dtype is: float64
    cloumn is: revenue ,Null values are: 0 , dtype is: float64

```python
# but we will convert them into string values 
df.fillna('Unknown',inplace = True)
```

```python
# to make the popularity rate more readable
df['popularity'] = df.popularity.round(2)
df.head(1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>title</th>
      <th>cast</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
  </tbody>
</table>
</div>

```python
# the generes, cast and production_companies are seperated wity | and can not reach the data easily
# so lets covert these columns into list of strings
df['genres'] = df['genres'].str.split('|')
df['cast'] = df['cast'].str.split('|')
df['production_companies'] = df['production_companies'].str.split('|')
```

```python
# now we need the main super star and the main production company and renamin their columns
df['cast'] = df['cast'].apply(lambda x: x[0])
df.rename(columns={'cast':'super_star'},inplace=True)

df['production_companies'] = df['production_companies'].apply(lambda x: x[0])
df.rename(columns={'production_companies':'production_companie'},inplace=True)
```

```python
# explodeing genres to be easy to deal with the different genres fo the same column
df_ex = df.explode('genres')
```

```python
df_ex.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>title</th>
      <th>super_star</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companie</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Adventure</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Science Fiction</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Thriller</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.42</td>
      <td>Mad Max: Fury Road</td>
      <td>Tom Hardy</td>
      <td>George Miller</td>
      <td>120</td>
      <td>Action</td>
      <td>Village Roadshow Pictures</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>3.481613e+08</td>
    </tr>
  </tbody>
</table>
</div>

Now we cleared and specified data and ready for the next step.

`<a id='eda'></a>`

## Exploratory Data Analysis

> In this section we will move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section.

### Q1 What are the most three genre produced??

> The first question make us able to know the distribution of the genres, in my openion it's important to know what is the most needed genre, which genre is not the best choice if i need to make a new movie and answering many question.
>
> To answer this question we need first to neglect the movies that has unkown genres, its ok we have a wide range of movies so a hundred movies will not affect then we need to count the movies for each genre then plot them.

```python
# first extract the data that movie genre is known
known_df = df_ex[df_ex['genres']!= 'Unknown']
```

```python
# function to calculate the mean of y grouped by x in the known_df
def df_col(x,y):
    return known_df.groupby(x)[y].mean()
```

```python
# getting the count of the genres
genres = known_df['genres'].value_counts()
genres
```

    Drama              4761
    Comedy             3793
    Thriller           2908
    Action             2385
    Romance            1712
    Horror             1637
    Adventure          1471
    Crime              1355
    Family             1231
    Science Fiction    1230
    Fantasy             916
    Mystery             810
    Animation           699
    Documentary         520
    Music               408
    History             334
    War                 270
    Foreign             188
    TV Movie            167
    Western             165
    Name: genres, dtype: int64

```python
plt.figure(figsize=(10,5))
plt.bar(genres.index, genres.values)

# to write the values of each movies genres count
def coordinates():
    for x,y in zip(genres.index,genres.values):
        label = "{:.1f}".format(y)
        plt.annotate(label, # this is the text
             (x,y), # these are the coordinates to position the label
             textcoords="offset points", # how to position the text
             xytext=(0,10), # distance from text to points (x,y)
             ha='center') # horizontal alignment can be left, right or center
coordinates()
  
plt.title('Count of Each Genres',fontname = 'monospace',fontsize=20)
plt.xlabel('Genre',fontname = 'monospace',fontsize=15)
plt.ylabel('Count',fontname = 'monospace',fontsize=15)

plt.tick_params(rotation = 90)
plt.grid(alpha=0.3,)
plt.show()
```

![png](output_30_0.png)

### Q2  How does movie genre and run time affects on movies rate?

> The second question make us see the correlation between the average rate of each genre and each genre charactrestic like runtime.
>
> To answer this question we have to get the average of rates and runtime for each genre the plot them

```python
# getting the average for each popularity and runtime
avg_rate = df_col('genres','popularity')
avg_run = df_col('genres','runtime')

# plotting them
plt.figure(figsize=(10,5))
plt.bar(avg_rate.index,avg_rate.values,alpha = 0.7,edgecolor='black')
plt.plot(avg_run.index,avg_run.values/80,alpha = 0.7,color='green',marker='o')

# to write the values of the Average Rate
for x,y in zip(avg_rate.index,avg_rate.values):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
         (x,y), # these are the coordinates to position the label
         textcoords="offset points", # how to position the text
         xytext=(0,-10), # distance from text to points (x,y)
         ha='center') # horizontal alignment can be left, right or center

# to write the values of the Average Run Time
for x,y in zip(avg_run.index,avg_run.values/80):
    label = "{:.1f}h".format(y*80/60) # to get the value in hour
    plt.annotate(label, # this is the text
         (x,y), # these are the coordinates to position the label
         textcoords="offset points", # how to position the text
         xytext=(0,10), # distance from text to points (x,y)
         ha='center') # horizontal alignment can be left, right or center

plt.xlabel('Genre',fontname = 'monospace',fontsize=15)
plt.ylabel('Average Rate',fontname = 'monospace',fontsize=15)

# To rotate the X axis genres
plt.tick_params(rotation =90)
plt.legend(['Avg Run Time','Average Rate'])

# To remove top and right spines
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

plt.grid(alpha=0.2)
plt.title('Average Rate For Each Genres With Runtime',fontname = 'monospace',fontsize=20)
plt.show()
```

![png](output_32_0.png)

### Q3 What are the most and the lowest genres the dirctors like to work on?

> Also directors may have their effect in this indusrty and may be the reason for attracting more viewrs to the movie
>
> So this question may be answered in many way in my case I prefered to get the number of the directors for each genre then we can easily choose which genere to work in and the directors in this genre that already achieved a good rate.

```python
# Knowing the number of directors for each genre
dir_genres= known_df.groupby('genres').director.nunique()
dir_genres= dir_genres.sort_values(ascending=False)
```

```python
# to make a gredient of color we need each color code
cust_color = ['#afddfa',
'#aad8f5',
'#a5d3ef',
'#a0cfea',
'#9ccae5',
'#97c5df',
'#92c0da',
'#8dbcd5',
'#89b7d0',
'#84b2ca',
'#7faec5',
'#7ba9c0',
'#76a5bb',
'#72a0b6',
'#6d9bb1',
'#6997ac',
'#6492a7',
'#608ea2',
'#5b899d',
'#578598',]
```

```python
plt.figure(figsize=(10,10))
plt.pie(dir_genres.values, labels=None, autopct='%1.1f%%', colors=cust_color, explode = [0.025 for i in range(len(cust_color))])
plt.title('% Of Directors For Each Genre',fontsize=20)
plt.legend(dir_genres.index, loc='center right', bbox_to_anchor=(1.2,0.5), title='Colors Legend')
plt.show()
```

![png](output_36_0.png)

### Q4 How does each genre cost and affect on the revenue?

> This question is very important, to know which genre takes high budget and gains an excelent revenue is important for each investor.

```python
# knowing the average of the budget and the revenue for each genre
plt.figure(figsize=(10,5))
rev_genre= df_col('genres','revenue')
budget_genre= df_col('genres','budget')

plt.plot(rev_genre.index,rev_genre.values,marker='o',alpha=0.5)
plt.plot(budget_genre.index,budget_genre.values,marker='o',color='green',alpha=0.5)

plt.xlabel('Genre',fontname = 'monospace',fontsize=15)
plt.ylabel('$ by Billion',fontname = 'monospace',fontsize=15)

plt.tick_params(rotation =90)
plt.legend(['Revenue','Budget'])

plt.title('Bugdet VS Revenue For Each Genre',fontname = 'monospace',fontsize=20)
plt.show()
```

![png](output_38_0.png)

### Q5 What is the the most produced genre in the last year and 1990?

> This question is to know how does the movie taste changed in the last 25 year and is it can change in future or not, by making comparsion between the count of movies in 2015 and 1990.

```python
# to get every count for every genre for the years years
# first we need to get the last year and 1990
years = known_df['release_year'].sort_values(ascending=False).unique()
years = years.tolist()
last_years = []
last_years.append(years[0])
last_years.append(years[years.index(1990)])
last_years

```

    [2015, 1990]

```python
# now we need to get the data for years
last_genre = known_df[known_df['release_year'].isin(last_years)]
last_genre.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>title</th>
      <th>super_star</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companie</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Adventure</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Science Fiction</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Thriller</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.42</td>
      <td>Mad Max: Fury Road</td>
      <td>Tom Hardy</td>
      <td>George Miller</td>
      <td>120</td>
      <td>Action</td>
      <td>Village Roadshow Pictures</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>3.481613e+08</td>
    </tr>
  </tbody>
</table>
</div>

```python
# now we have get the count of genres for 2015
genre_2015= last_genre[last_genre['release_year']==last_years[0]]
genre_2015= genre_2015['genres'].value_counts()
genre_2015
```

    Drama              260
    Thriller           171
    Comedy             162
    Horror             125
    Action             107
    Science Fiction     86
    Adventure           69
    Romance             57
    Documentary         57
    Crime               51
    Family              44
    Mystery             42
    Animation           39
    Music               33
    Fantasy             33
    TV Movie            20
    History             15
    War                  9
    Western              6
    Name: genres, dtype: int64

```python
# now we have get the count of genres for 1990
genre_1990= last_genre[last_genre['release_year']==last_years[1]]
genre_1990= genre_1990['genres'].value_counts()
genre_1990
```

    Drama              60
    Comedy             48
    Thriller           46
    Action             39
    Crime              30
    Horror             26
    Adventure          23
    Romance            19
    Science Fiction    18
    Mystery            14
    Fantasy            13
    Family             12
    History             4
    Animation           4
    Western             3
    Music               2
    War                 2
    Foreign             1
    TV Movie            1
    Documentary         1
    Name: genres, dtype: int64

```python
plt.figure(figsize=(10,10))
plt.bar(genre_2015.index,genre_2015.values,alpha = 0.5, edgecolor='black')
plt.bar(genre_1990.index,genre_1990.values,alpha = 0.5, color = 'green', edgecolor='black')

plt.xlabel('Genre',fontname = 'monospace',fontsize=15)
plt.ylabel('Count',fontname = 'monospace',fontsize=15)

# To rotate the X axis genres
plt.tick_params(rotation =90)
plt.legend(['2015','1990'])

# To remove top and right spines
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

plt.grid(alpha=0.2)
plt.title('Count Of Movies For Each Genre For 2015 & 1990',fontname = 'monospace',fontsize=20)
plt.show()
```

![png](output_44_0.png)

## Q6 what is the relation between movie time and the budget?

> Here a question about movies characterstics budget and the runtime and is there a relation between them or now.

```python
# before we plot and answer this question we first need to make runtime more readable 
df.head(1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>title</th>
      <th>super_star</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companie</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>[Action, Adventure, Science Fiction, Thriller]</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
  </tbody>
</table>
</div>

```python
# first we need to make groups for each hour in new list
runtime = []
for i in df.runtime:
    if i <= 60:
        runtime.append('1 Hour')
    elif 60 < i <= 120:
        runtime.append('2 Hours')
    elif 120 < i <= 180:
        runtime.append('3 Hours')
    elif i > 180:
        runtime.append('4+ Hours')
```

```python
# now we need to make new column contains these groups
df['runtime_groups'] = runtime
df.head(1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>title</th>
      <th>super_star</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companie</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime_groups</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.99</td>
      <td>Jurassic World</td>
      <td>Chris Pratt</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>[Action, Adventure, Science Fiction, Thriller]</td>
      <td>Universal Studios</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
      <td>3 Hours</td>
    </tr>
  </tbody>
</table>
</div>

```python
# because we grouped the runtime
# lets get the average budget for each group of runtime
average_buget_time = df.groupby('runtime_groups')['budget'].mean()
```

```python
plt.figure(figsize=(10,5))
plt.plot(average_buget_time.index,average_buget_time.values)
plt.xlabel('Hour Group',fontname = 'monospace',fontsize=15)
plt.ylabel('Avg. $ By Million',fontname = 'monospace',fontsize=15)
plt.title('Average Budget Per Movie Time',fontname = 'monospace',fontsize=20)
plt.show()
```

![png](output_50_0.png)

#### Another plot shows us the realtion for each movie run time and the budget to make the vision more clear.

```python
# Scatter plot figure shows the relation between all move runtime and their budget
plt.figure(figsize=(10,10))
plt.scatter(df.runtime.values/60,df.budget.values ,alpha=0.5)
plt.xlabel('Hours',fontname = 'monospace',fontsize=15)
plt.ylabel('$ By Million',fontname = 'monospace',fontsize=15)
plt.title('Budget Per Movie Time',fontname = 'monospace',fontsize=20)
plt.show()
```

![png](output_52_0.png)

`<a id='conclusions'></a>`

## Conclusions

**Q1 What are the most three genres produced?**

> * The most three qenres produced are *drama, comedy, thriller*.

**Q2 How does movie genre and run time affects on movies rate?**

> * Its obvious that the genres the have an average runtime is around 1.5 hours have the higher rate like *adventure, fantasy, science fiction*.
> * Also low run average time was very useful in *animation genre* with that has high average rating.
> * On the other hand the genres with high runtime over 2 hours in average have medium rate like *history and war genres*.

**Q3 What are the most and the lowest genres the dirctors like to work on?**

> * The most genres the directos works on are the most genres produced in Q1 *drama, comedy, thriller*, and these three genre has an average rate higher than the medium, maybe means that these genres are the safe zone for the directors.
> * The lowest genres the directos works on are *western, tv movie, and foreign*, altough the foreign genre has a medium average rate.

**Q4 How does each genre cost and affect on the revenue?**

> * It's obvious that *adventure, fantasy and science fiction* from Q2 have medium average run time have also the higer cost and the higher revenue.
> * On the other hand *documentary, foreign and tv movies* have the lowest average cost and approximately no revenue.

**Q5 What is the the most produced genre in the last year and 1990?**

> * The most produced genre in 2015 and 1990 are *drama, thriller and comedy* the taste doesn't change alot but the difference of the number of movies generated in these years is huge for example drama 1990 produced around 60 movie but in 2015 produced 260 movie, approximately 200 movie.
> * We can also see that *documentary, music, animation and tv movies* counts in 1990 wasn't exceed 10 movies.

**Q6 what is the relation between movie time and the budget?**

> * We can find that in general movies around 3 hours runtime costs alot in average.
> * But in details from one to 4 hours are the most expensive movies specially 2 hours movies ofcourse the cost varies depending on other characteristics like the genre, but the runtime around 2 hours have a huge variaty of budgets.
