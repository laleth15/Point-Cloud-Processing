def plane_fit_RANSAC(points, num_iterations, distance_threshold, min_inliers):
    """
    Fits a plane to a set of points using the RANSAC algorithm.
    
    Args:
    points: A 2D numpy array of shape (n, 3), where n is the number of points and
    each row contains the (x, y, z) coordinates of a point.
    num_iterations: The number of iterations to run the RANSAC algorithm.
    distance_threshold: The maximum distance between a point and the plane for it to be considered an inlier.
    min_inliers: The minimum number of inliers required for a plane to be considered a good fit.
    
    Returns:
    A tuple (a, b, c, d) representing the coefficients of the best-fit plane in the general
    equation ax + by + cz + d = 0.
    """
    # Initialize variables for the best plane found so far
    most_inliers = 0
    best_plane = None
    
    for i in range(num_iterations):
        # Randomly sample 3 points from the input data
        sample = points[np.random.choice(points.shape[0], 3, replace=False), :]
        
        # Fit a plane to the sample using least square fit method
        plane = least_sq_fit(sample)
        
        # Calculate the distance between each point and the plane
    distances = np.abs(np.dot(points - np.array([plane]), normal))
    
    # Count the number of inliers
    inliers = distances < distance_threshold
    num_inliers = np.count_nonzero(inliers)
    
    # Update the best plane if this iteration produced a better fit
    if num_inliers > most_inliers:
        most_inliers = num_inliers
        best_plane = plane
        
        # If enough inliers are found, refit the plane using all inliers
        if num_inliers > min_inliers:
            best_plane = least_sq_fit(points[inliers])
            
    # Update iteration counter
    iteration += 1
    
    # Return the best-fit plane
    return best_plane

#Load the data and plot the point cloud

data = np.loadtxt('points.txt')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#Find the plane that best fits the data using RANSAC

num_iterations = 1000
distance_threshold = 0.05
min_inliers = 100
best_plane = plane_fit_RANSAC(data, num_iterations, distance_threshold, min_inliers)

#Print the coefficients of the best-fit plane

print('Best-fit plane coefficients: a = {:.4f}, b = {:.4f}, c = {:.4f}, d = {:.4f}'.format(*best_plane))

#Plot the best-fit plane

xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
zz = (-best_plane[3] - best_plane[0]*xx - best_plane[1]*yy) / best_plane[2]
ax.plot_surface(xx, yy, zz, alpha=0.5)
plt.show()