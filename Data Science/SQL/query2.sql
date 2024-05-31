set echo on
set timing on
column column_name format a15
set linesize 200
--------------------------------------------------------------------------------
--Query:MAX movies by an actor in years 1962-1971
--------------------------------------------------------------------------------
SELECT NCONST||','||primaryname||','||startyear||','||counter
FROM (SELECT NB.NCONST,NB.primaryname,TB.startyear,COUNT(*) AS counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor') AND TB.startyear NOT LIKE '\N' AND
        TB.startyear > '1961' AND
        TB.startyear < '1972'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear)
WHERE counter = (SELECT MAX(counter) 
FROM (SELECT NB.primaryname,TB.startyear,COUNT(*) AS counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor') AND TB.startyear NOT LIKE '\N' AND
        TB.startyear > '1961' AND
        TB.startyear < '1972'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear));
--------------------------------------------------------------------------------
--Query:MAX movies by an actress in years 1962-1971
--------------------------------------------------------------------------------
SELECT NCONST||','||primaryname||','||startyear||','||counter
FROM (SELECT NB.NCONST,NB.primaryname,TB.startyear,COUNT(*) AS counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actress') AND TB.startyear NOT LIKE '\N' AND
        TB.startyear > '1961' AND
        TB.startyear < '1972'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear)
WHERE counter = (SELECT MAX(counter) 
FROM (SELECT NB.primaryname,TB.startyear,COUNT(*) AS counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actress') AND TB.startyear NOT LIKE '\N' AND
        TB.startyear > '1961' AND
        TB.startyear < '1972'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear));
--------------------------------------------------------------------------------
--Query:MIN 3 movies by an actor in years 1962-1971
--------------------------------------------------------------------------------
SELECT NB.NCONST||'.'||NB.primaryname||','||TB.startyear||','||COUNT(*)
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
HAVING TB.startyear > '1961' AND
        TB.startyear < '1972' AND
        COUNT(*) = 3
ORDER BY TB.startyear;
--------------------------------------------------------------------------------
--Query:MIN 3 movies by an actress in years 1962-1971
--------------------------------------------------------------------------------
SELECT NB.NCONST||'.'||NB.primaryname||','||TB.startyear||','||COUNT(*)
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actress') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
HAVING TB.startyear > '1961' AND
        TB.startyear < '1972' AND
        COUNT(*) = 3
ORDER BY TB.startyear;
set echo off