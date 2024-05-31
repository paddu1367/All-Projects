set echo on
set timing on
column column_name format a15
set linesize 200
--------------------------------------------------------------------------------
--Query: to find the NCONST of an actor,actress giving the name
--------------------------------------------------------------------------------
SELECT NCONST,primaryname
FROM imdb00.name_basics 
where (Lower(primaryname) LIKE '%james cagney%' OR Lower(primaryname) LIKE '%halle berry%');

--------------------------------------------------------------------------------
-- query: list number of movies per year using actor nameid of 'james cagney' over their career
--------------------------------------------------------------------------------

SELECT NB.primaryname||','||TB.startyear||','||COUNT(*)
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        NB.nconst = 'nm0000010' AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear;
--------------------------------------------------------------------------------
---- query: list number of movies per year using actress nameid of 'halle berry' over their career
--------------------------------------------------------------------------------

SELECT NB.primaryname||','||TB.startyear||','||COUNT(*)
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        NB.nconst = 'nm0000932' AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actress') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear;
--------------------------------------------------------------------------------
-- query: list number of movies per year using actor primaryname of 'chris hemsworth' over their career
--------------------------------------------------------------------------------

SELECT NB.primaryname||','||TB.startyear||','||COUNT(*)
FROM imdb00.name_basics NB, imdb00.title_principals TP, imdb00.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        Lower(NB.primaryname) LIKE 'chris hemsworth' AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actor') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear;

---------------------------------------------------------------------------------
--query: list number of movies per year using actress primaryname of 'natalie portman' over their career
-------------------------------------------------------------------------------

SELECT NB.primaryname||','||TB.startyear||','||COUNT(*)
FROM sharmac.name_basics NB, sharmac.title_principals TP, sharmac.title_basics TB
WHERE NB.nconst = TP.nconst AND 
        TP.tconst = TB.tconst AND 
        Lower(NB.primaryname) LIKE 'natalie portman' AND
        LOWER(TB.titleType) = 'movie' AND
        LOWER(TP.category) IN ('actress') AND TB.startyear NOT LIKE '\N'
GROUP BY NB.nconst, NB.primaryname, TB.startyear
ORDER BY TB.startyear;

Set echo off

