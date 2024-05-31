set echo on
set timing on
column column_name format a15
set linesize 200
--------------------------------------------------------------------------------
--Query:to split acting years in to span between 1930-1953
--------------------------------------------------------------------------------
SELECT primaryname||','||SUM(Counter) FROM (
SELECT NB.primaryname,TB.startyear,COUNT(*) AS Counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        (NB.nconst = 'nm0000010' OR NB.nconst = 'nm0000932' OR 
        Lower(primaryname) LIKE '%chris hemsworth%' OR
        Lower(NB.primaryname) LIKE '%natalie portman%' )AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor','actress') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
HAVING TB.startyear > '1929' AND
        TB.startyear < '1954'
ORDER BY TB.startyear)
GROUP BY primaryname;
--------------------------------------------------------------------------------
--Query:to split acting years in to span between 1954-1977
--------------------------------------------------------------------------------
SELECT primaryname||','||SUM(Counter) FROM (
SELECT NB.primaryname,TB.startyear,COUNT(*) AS Counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        (NB.nconst = 'nm0000010' OR NB.nconst = 'nm0000932' OR 
        Lower(primaryname) LIKE '%chris hemsworth%' OR
        Lower(NB.primaryname) LIKE '%natalie portman%' )AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor','actress') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
HAVING TB.startyear > '1953' AND
        TB.startyear < '1978'
ORDER BY TB.startyear)
GROUP BY primaryname;
--------------------------------------------------------------------------------
--Query:to split acting years in to span between 1978-2001
--------------------------------------------------------------------------------
SELECT primaryname||','||SUM(Counter) FROM (
SELECT NB.primaryname,TB.startyear,COUNT(*) AS Counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        (NB.nconst = 'nm0000010' OR NB.nconst = 'nm0000932' OR 
        Lower(primaryname) LIKE '%chris hemsworth%' OR
        Lower(NB.primaryname) LIKE '%natalie portman%' )AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor','actress') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
HAVING TB.startyear > '1977' AND
        TB.startyear < '2002'
ORDER BY TB.startyear)
GROUP BY primaryname;
--------------------------------------------------------------------------------
--Query:to split acting years in to span between 2002-2024
--------------------------------------------------------------------------------
SELECT primaryname||','||SUM(Counter) FROM (
SELECT NB.primaryname,TB.startyear,COUNT(*) AS Counter
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        (NB.nconst = 'nm0000010' OR NB.nconst = 'nm0000932' OR 
        Lower(primaryname) LIKE '%chris hemsworth%' OR
        Lower(NB.primaryname) LIKE '%natalie portman%' )AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor','actress') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
HAVING TB.startyear > '2002' AND
        TB.startyear < '2024'
ORDER BY TB.startyear)
GROUP BY primaryname;

set echo OFF
