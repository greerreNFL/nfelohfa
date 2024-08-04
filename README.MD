# NFELO HFA
## About
nfelohfa is a package for projecting home field advantage (HFA) at the individual game level

Estimates are stored in the estimated_hfa.csv in the root folder of the package. This file uses game_id as it's row identifier for easy joining to other nfelo and, to a degree, nflfastR datasets. Nfelo id's use constant values for team abbreviation, so to join to nflfastR, you would need to do replacement where appropriate (like LV for OAK, SD for LAC, STL for LAR)

The model leverages two frameworks for making individual game estiamtes:
* HFA Base - the base model that estimates _league_ wide HFA overtime. This is LOESS-like in that it uses a rolling regression of past week's observed HFA to make a projection of the current week's HFA. To smooth out these projections, it is then passed through an exponential moving average (EMA)
* HFA Adj - individual adjustments to HFA Base are then made at the _game_ level based on meta data context like bye weeks, timezone differences, field surface, and matchup type

To read more about the framework, visit [A Further Exploration of Home Field Advantage in the NFL](https://www.nfeloapp.com/analysis/a-further-exploration-of-home-field-advantage-in-the-nfl/)

## Performance
The model is optimized for mean absolute error (MAE) of projecting final scores using SRS based ratings for home and away teams. Said another way The error is the difference between the actual result and the result we would project by adding estimated HFA to the home team's rating and subtracting the away team's rating.

Overall the lift in accuracy is minimal versus using a standard plug value for HFA like 2.5. While HFA has decreased overtime, and that is reflected in HFA Base, the magnitude of the decrease is fairly small (.5 to 1 points), and observed HFA is very noisy. Together, that makes it difficult for the HFA Base to provide any lift at all. Demonstrably all the lift in this model is created by the adjustments.