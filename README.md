# Point-Cloud-Processing
In this project we will process point clouds for plane, sphere and cylinder fitting using RANSAC. And implement Iterative Closest Point to perform scan matching on two point clouds.

To Run the code:

# Plane fitting
1. By calculating the sample mean and covariancematrix of the points.
2. With outliers.
3. Using RANSAC and outliers.

python point_cloud.py -q <q1_a,q1_b,q1_c>

# Sphere fitting using RANSAC

python point_cloud.py -q q2

# Cylinder fitting using RANSAC


python point_cloud.py -q q3

# You can run the scan matching program by running it in your IDE.

