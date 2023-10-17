ECA_July_2023
-------------
1.0 Explanation on vehicle_mpg.tsv & vehicle_manufacturers.csv
---------------------------------------------------------------
Column definitions are stated in the respective file's header



2.0 Explanation on mov_genre.dat, mov_item.dat, mov_rating.dat, mov_occupation.dat, mov_user.dat
-------------------------------------------------------------------------------------------------

2.1 mov_genre.dat
- movie genre (1st column), genre identifier (2nd column)

2.2 mov_rating.dat
- user/reviewer identifier (1st column), movie identifier (2nd column), 
   rating (3rd column), timestamp (4th column)

2.3 mov_item.dat
- movie identifier (1st column), movie name (2nd column), 
    release date (3rd column), video release date (4th column),
    IMDb URL (5th column), movie genre list* (6th column till 24th column)

*movie genre list = unknown | Action | Adventure | Animation |
                       Children's | Comedy | Crime | Documentary | 
                       Drama | Fantasy| Film-Noir | Horror | Musical | 
                       Mystery | Romance | Sci-Fi | Thriller | War | Western |

2.4 mov_occupation.dat
- occupation name (1st column)

2.5 mov_user.dat
- user/reviewer identifier (1st column), age (2nd column), gender (3rd column),
     occupation (4th column), zip code (5th column) 
