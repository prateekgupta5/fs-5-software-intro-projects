import matplotlib.pyplot as plt
import numpy as np
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

#updating the graph takes the most compute
#if we are simply generationg data it is not immediately neccicarry
#thus we toggle it here
DO_GRAPH = False

#name of the file where the data will be stored.
OUTFILE = "training_data.npz"

K_P = 0.22
K_I = 0 # 0.01 #zeroed for ai/ml
K_D = 0 # 0.01 #zeroed for ai/ml

#number of times we change desired_v
NUM_TARGET_VELOCITY_ITERS = 100

#number of simulation steps per desired_v
STEPS = 400

NUM_DATAPTS = NUM_TARGET_VELOCITY_ITERS*STEPS

#current datapt index
datapt = 0

#batch the real-time graph updates for better simulation preformance
GRAPH_BATCH_SIZE = 100

car = make_car(desired_v=20.0, dt=0.1)

#buffers to store the generated data.
#because we know how much data we will generate, we can allocate it ahead of time
recorded_error        = np.zeros(NUM_DATAPTS, dtype = np.float32)
recorded_velocity     = np.zeros(NUM_DATAPTS, dtype = np.float32)
recorded_target       = np.zeros(NUM_DATAPTS, dtype = np.float32)
recorded_acceleration = np.zeros(NUM_DATAPTS, dtype = np.float32)

x_axis = np.linspace(1, NUM_DATAPTS, num=NUM_DATAPTS, dtype=int)

if(DO_GRAPH):
  # Add the human-readable parts of the plot
  plt.figure(figsize=(7, 5))
  plt.title("PID Controller Output Over Time")
  plt.xlabel("Steps")
  plt.ion()
  plt.show()

  #initialise the lines and then update the legend
  plt.plot([0], [0], color='black' , linewidth=1, label='Target Velocity (m/s)'               )
  plt.plot([0], [0], color='blue'  , linewidth=2, label='Velocity Over Time (m/s)'            )
  plt.plot([0], [0], color='red'   , linewidth=2, label='Velocity Error Over Time (m/s)'      )
  plt.plot([0], [0], color='yellow', linewidth=2, label='Acceleration Over Time (m/s^2)')

  plt.legend()

for velocity in range(NUM_TARGET_VELOCITY_ITERS):
  car["desired_v"] = np.random.uniform(2, 50)
  for step in range(STEPS):
    
    #run pid controler to get error and theoreticaly desired acceleration
    (acceleration, error) = calculate_desired_acceleration(car, K_P, K_I, K_D)
      
    #convert the theoretically desired acceleration to a feasable
    #  percentage of the maximum power of the motor 
    throttle_percentage = acceleration_to_throttle_percentage(acceleration)
    
    #send the desired power to the car and update the simulation
    update(car, throttle_percentage)

    #log the recorded values
    recorded_target[datapt] = (car["desired_v"])
    recorded_error[datapt] = (error)
    recorded_velocity[datapt] = (car["v"])
    recorded_acceleration[datapt] = (acceleration)

    #every GRAPH_BATCH_SIZE steps, graph: <velocty>, <error>, <target velocity>, <aceleration>
    if DO_GRAPH and (step % GRAPH_BATCH_SIZE == 0):
      plt.plot(x_axis[:datapt - 1], recorded_target      [:datapt - 1], color='black' , linewidth=1, label='Target Velocity Time (m/s)'     )
      plt.plot(x_axis[:datapt - 1], recorded_velocity    [:datapt - 1], color='blue'  , linewidth=2, label='Velocity Over Time (m/s)'       )
      plt.plot(x_axis[:datapt - 1], recorded_error       [:datapt - 1], color='red'   , linewidth=2, label='Velocity Error Over Time (m/s)' )
      plt.plot(x_axis[:datapt - 1], recorded_acceleration[:datapt - 1], color='yellow', linewidth=2, label='Acceleration Over Time (m/s^2)' )
      print(error)
      
      #pause graph until next update
      plt.pause(0.01)

    #move on to next datapt
    datapt += 1

#At this point the data is ordered over time. Not only might this
#   have an impact on training, (I am not certain though) but it
#   also would introuduce a bias into a training/testing split if
#   we simply split by index.
#Thus, we first shuffle all the arrays. As we still need the collums 
#   to line up, we shuffle the indexes instead.

print("Shuffling training data")

#buffers to hold final data
recorded_velocity_     = np.zeros(NUM_DATAPTS)
recorded_error_        = np.zeros(NUM_DATAPTS)
recorded_target_       = np.zeros(NUM_DATAPTS)
recorded_acceleration_ = np.zeros(NUM_DATAPTS)

#shuffled indexes
shuffled_indexes = np.random.permutation(NUM_DATAPTS)

for i in range(NUM_DATAPTS):
    recorded_velocity_    [i] = recorded_velocity    [shuffled_indexes[i]]
    recorded_error_       [i] = recorded_error       [shuffled_indexes[i]]
    recorded_target_      [i] = recorded_target      [shuffled_indexes[i]]
    recorded_acceleration_[i] = recorded_acceleration[shuffled_indexes[i]]

#save the generated data in read-efficient file format (labled ndarray data dump)
np.savez(OUTFILE,
         recorded_target=recorded_target_,
         recorded_error=recorded_error_,
         recorded_velocity=recorded_velocity_,
         recorded_acceleration=recorded_acceleration_
)

if DO_GRAPH:
  input()