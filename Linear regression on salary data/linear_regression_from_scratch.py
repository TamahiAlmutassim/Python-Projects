import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def mean_squared_error(b,m,coordinates):

	x = coordinates[:, 0] #Get all the x points
	y = coordinates[:, 1] #Get all the y points 
	error = y - (m * x + b)

    #getting the average 
	return float(np.mean(error ** 2))

def gradient_descent_optimization(points,starting_b,starting_m,learning_rate,num_iteration):
	#star ting valuer for b,m
	b=starting_b
	m=starting_m
	MSE_per_epoch=[] #To store the means squared error 
	#gradient sedcent
	line_history = []
	for i in range(num_iteration):
		#updating b and m with more accurate b , m 
		b, m= step_gradiant(b, m, points, learning_rate)
		# compute error AFTER update
		error = mean_squared_error(b, m, points)
		MSE_per_epoch.append(error)
		line_history.append((b, m))
	return b, m, MSE_per_epoch, line_history

def step_gradiant(b_current,m_current,points,LearnignRate):
	#starting points for the gradient
	b_gradient=0
	m_gradient=0
	n=float(len(points))
	for i in range(0,len(points)):
		x=points[i,0]
		y=points[i,1]
		#calculating the gradient descent
		b_gradient+=-(2/n)*(y-(m_current*x+b_current))
		m_gradient+=-(2/n)* x*(y-(m_current*x+b_current))
	#updating or b and m values using partial derivatives
	new_b=b_current-(LearnignRate*b_gradient)
	new_m=m_current-(LearnignRate*m_gradient)
	return new_b,new_m




def run():
	#first we get our data 
	coordinates = np.genfromtxt('dataset/Salary_Data.csv', delimiter=',', skip_header=1)

	#step 2: Define hyperparameters of the learning process
	learning_rate=0.01               # How fast our mode converge 
	#y=mx+b (the slope equation , where y is our dependant variable and x is the independent variable)
	initial_b=0
	initial_m=0
	epochs=1000 #The number of times our learning process going to repeat itself 

	#step 3 : The training process
	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(
    initial_b,
    initial_m,
	mean_squared_error(initial_b, initial_m, coordinates)))
	b, m, MSE_per_epoch, line_history=gradient_descent_optimization(coordinates,initial_b,initial_m,learning_rate,epochs)
	print("Ending point at b = {1}, m = {2}, error = {3}".format(
    epochs,
    b,
    m, 
	mean_squared_error(b, m, coordinates)))

	#To make sure our work is correct we are going to compare the results using scikit-learn
	x = coordinates[:, 0].reshape(-1, 1)
	y = coordinates[:, 1]

	model = LinearRegression()
	model.fit(x, y)
	print('This is working')
	print("Sklearn m:", model.coef_[0])
	print("Sklearn b:", model.intercept_)



	#plotting 
# ===== PLOT 1: Regression line =====
	x = coordinates[:, 0]
	y = coordinates[:, 1]

	# sort x for clean line
	sorted_indices = np.argsort(x)

	plt.scatter(x, y)
	plt.plot(x[sorted_indices], (m * x + b)[sorted_indices])

	plt.xlabel("Years of Experience")
	plt.ylabel("Salary")
	plt.title("Salary vs Experience")

	plt.show()


	# ===== PLOT 2: Loss curve =====
	plt.figure()
	plt.plot(range(len(MSE_per_epoch)), MSE_per_epoch)

	plt.xlabel("Epochs")
	plt.ylabel("Mean Squared Error")
	plt.title("Loss Curve (MSE vs Epochs)")

	plt.show()
	return coordinates, line_history



if __name__ == "__main__":
    coordinates, line_history = run()

    import matplotlib.animation as animation

    fig, ax = plt.subplots()

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    ax.scatter(x, y)

    line, = ax.plot([], [], color="red")

    def update(frame):
        b, m = line_history[frame]   

        y_pred = m * x + b
        line.set_data(x, y_pred)

        return line,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(line_history),
        interval=50,
        repeat=False   
    )

    plt.title("Gradient Descent Line Fitting Animation")
    plt.show()