# K-mean-Elkan
a k-means algorithm which employs accelerated Elkan method and can use different distance functions

Three data set are provided for evaluation as follow with the following names:
Birch: X_birch_m.txt, y_birch_m.txt
KDDup: X_cup.txt, y_cup.txt

To run the program you can type:
python Q3_Elkan argumnets

results will be written displayed on the screen

Arguments for the command should be provided after these keys (some of them are optional:

-itr1  : name of a file containing attribute values
-itr2   : name of a file containing class values

OPTIONALS:
-its1  : name of a file containing attribute values
-its2   : name of a file containing class values
-i    : itteration number for hard stoppping the program(DEFAULT: 34)
-K	: number of desired cluster (DEFAULT: 3)

sample running commands:

python K_means_Elkan.py -itr1 X_birch_m.txt -itr2 y_birch_m.txt -i 34 -K 3 

python K_means_Elkan.py -itr1 X_birch_m.txt -itr2 y_birch_m.txt -i 325 -K 100 
