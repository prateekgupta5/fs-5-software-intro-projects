import matplotlib.pyplot as plt
import numpy as np
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 0.1
K_I = 0.1
K_D = 0.1
 
STEPS = 550 
car = make_car(desired_v=20.0, dt=0.1)

recorded_error = []
recorded_velocity  = []

# Add the human-readable parts of the plot
plt.figure(figsize=(7, 5))
plt.title("P Controller Output Over Time")
plt.xlabel("Steps")
plt.ion()
plt.show()

plt.plot([0], [0], color='blue', linewidth=2, label='Velocity Over Time (m/s)'       )
plt.plot([0], [0], recorded_error   , color='red' , linewidth=2, label='Velocity Error Over Time (m/s)' )
plt.legend()

for step in range(STEPS):
  
  #run p controler to get error and theoreticaly desired acceleration
  (acceleration, error) = calculate_desired_acceleration(car, 0.9)
    
  #convert the theoretically desired acceleration to a feasable
  #  percentage of the maximum power of the motor 
  throttle_percentage = acceleration_to_throttle_percentage(acceleration)
  
  #send the desired power to the car and update the simulation
  update(car, throttle_percentage)
  step = step + 1 #step the simulation

  #log the recorded values
  recorded_error.append(error)
  recorded_velocity.append(car["v"])

  #graph: <desired_value>, <error>
  plt.plot(np.linspace(1, step, num=step, dtype=int), recorded_velocity, color='blue', linewidth=2, label='Velocity Over Time (m/s)'       )
  plt.plot(np.linspace(1, step, num=step, dtype=int), recorded_error   , color='red' , linewidth=2, label='Velocity Error Over Time (m/s)' )
  plt.pause(0.05)

plt.show()
input("")