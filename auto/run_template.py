import matplotlib.pyplot as plt
import numpy as np
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 0.22
K_I = 0.01
K_D = 0.01
 
STEPS = 550
GRAPH_BATCH_SIZE = 20

car = make_car(desired_v=20.0, dt=0.1)

recorded_error = []
recorded_velocity  = []
x_axis = np.linspace(1, STEPS, num=STEPS, dtype=int)

# Add the human-readable parts of the plot
plt.figure(figsize=(7, 5))
plt.title("PI Controller Output Over Time")
plt.xlabel("Steps")
plt.ion()
plt.show()

#initialise the lines and then update the legend
plt.plot(np.linspace(1, STEPS, num=STEPS, dtype=int), [car["desired_v"]]*STEPS, color='black', linewidth=1, label='Target Velocity (m/s)'       )
plt.plot([0], [0], color='blue', linewidth=2, label='Velocity Over Time (m/s)'       )
plt.plot([0], [0], color='red' , linewidth=2, label='Velocity Error Over Time (m/s)' )
plt.legend()

for step in range(STEPS):
  
  #run p controler to get error and theoreticaly desired acceleration
  (acceleration, error) = calculate_desired_acceleration(car, K_P, K_I, K_D)
    
  #convert the theoretically desired acceleration to a feasable
  #  percentage of the maximum power of the motor 
  throttle_percentage = acceleration_to_throttle_percentage(acceleration)
  
  #send the desired power to the car and update the simulation
  update(car, throttle_percentage)
  step = step + 1 #step the simulation

  #log the recorded values
  recorded_error.append(error)
  recorded_velocity.append(car["v"])

  #ever GRAPH_BATCH_SIZE steps, graph: <desired_value>, <error>
  if step % GRAPH_BATCH_SIZE == 0:
    plt.plot(x_axis[:step], recorded_velocity, color='blue', linewidth=2, label='Velocity Over Time (m/s)'       )
    plt.plot(x_axis[:step], recorded_error   , color='red' , linewidth=2, label='Velocity Error Over Time (m/s)' )
    print(error)
    #pause graph until next update
    plt.pause(0.01)
input()