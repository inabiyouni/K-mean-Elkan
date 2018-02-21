# K-mean-Elkan
a k-means algorithm which employs accelerated Elkan method and can use different distance functions

To run the program type:

python K_means_Elkan.py -itr1 first_file_name -itr2 first_data_classes -its1 second_file_name -its2 second_data_classes -f dist_function -a visualization -K number_of_clusters

which:

first_file_name, second_file_name: name of the first data file which should include data points in rows and features in columns

first_data_classes, second_data_classes: name of the first class file which should include real classes for each data point

dist_function: name of the desired distance function which can be: "eucli" "cosine" "fun1" "fun2"

visualization: To turn visualization off or on by choosing: "false" " true" respectively (it is recommended that use this option for data with only two clusters

number_of_clusters: number of clusters which you need from data. If you skip this argument, the program with assign number of classes in the input class file

example:

python K_means_Elkan.py -itr1 X_iris.txt -itr2 y_iris.txt -f eucli -a false -K 3

python K_means_Elkan.py -itr1 X_data1.txt -itr2 y_data1.txt -its1 X_data2.txt -its2 y_data2.txt -f fun2 -a true