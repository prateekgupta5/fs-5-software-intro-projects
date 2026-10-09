import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from model import PID_Aproximator

#for final test
from generate_training_data import K_P
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

DO_PRINT       = True
DO_GRPAH_TRAIN = False
DO_FINAL_TEST  = True

#the quantity of data we wnt to use for training
PERCENTAGE_TRAINING_DATA = 0.8

LEARNING_RATE = 0.0002

EPOCHS = 1000

pid_data = np.load("training_data.npz")
num_datapts = len(pid_data["recorded_velocity"])

if(DO_PRINT):
  print("Loaded data")

#STATE_TENSOR = [{velocity over time}, {target over time}]
#   describes the state that would be inputed to the PID controller
#
#acceration_vector = [{unclipped acceleration over time}]
#   describes the desired change in state outputed by the PID controller
STATE_TENSOR = torch.tensor(
    np.stack((pid_data["recorded_velocity"], pid_data["recorded_target"]), axis=1),
    dtype=torch.float32
) #(N, 2)

acceleration_vector = torch.tensor(
    pid_data["recorded_acceleration"], dtype=torch.float32
).unsqueeze(1) #(N, 1)

if(DO_PRINT):
  print("Construced tensors")

# Create train/test split
tran_split = int(PERCENTAGE_TRAINING_DATA * acceleration_vector.numel())
STATE_train, accel_train = STATE_TENSOR[:tran_split], acceleration_vector[:tran_split]
STATE_test,  accel_test  = STATE_TENSOR[tran_split:], acceleration_vector[tran_split:]

if DO_PRINT:
  print("State train, State Test", STATE_train.shape, ",", STATE_test.shape)
  print("Accel train, Accel test", accel_train.shape, ",", accel_test.shape)

#Now we have usable training data and testing data

#instanciate model
approx = PID_Aproximator()

#define training tools
loss_fn = nn.L1Loss() #MAE Loss because the aproxximator is preforming linear regression
optimizer = torch.optim.SGD(params=approx.parameters(), lr=LEARNING_RATE)

#buffers to store data
train_losses = np.zeros(int(EPOCHS), np.float32)
test_losses  = np.zeros(int(EPOCHS), np.float32)
epoch_count  = np.arange(1, EPOCHS + 1, dtype = np.int32)

if(DO_PRINT):
  print("Beggening training")

for epoch in range(EPOCHS):
  #switch to training mode
  approx.train()

  #gather predictions and score them
  train_predictions = approx(STATE_train)
  loss = loss_fn(train_predictions, accel_train)

  #adjust model based on score
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  approx.eval()

  #test model with test data
  with torch.inference_mode():
    train_predictions = approx(STATE_test)
    test_loss = loss_fn(train_predictions, accel_test)

  #log training/testing data
  train_losses[epoch] = loss.item()
  test_losses[epoch] = test_loss.item()
  if epoch % 50 == 0:
    # print(f"epoch {epoch}: train {loss:.4f}, tests {test_loss:.4f}")

    # Print out what's happening
    if DO_PRINT:
      print(f"Epoch: {epoch:>4} | MAE Train Loss: {loss:>4.15f} | MAE Test Loss: {test_loss:>4.15f} ")

if DO_PRINT:
  print(approx.linear_layer.weight, approx.linear_layer.bias)

if DO_GRPAH_TRAIN:
  plt.figure(figsize=(7, 5))
  plt.title("Training/Testing Error over time")
  plt.xlabel("Epochs")
  plt.ion()

  #initialise the lines and then update the legend
  plt.plot(epoch_count, train_losses, color='blue'  , linewidth=2, label='Loss with Training Inputs')
  plt.plot(epoch_count, test_losses, color='red'   , linewidth=2, label='Loss with Testing Inputs'  )

  plt.legend()
  plt.show()

  input()

  plt.close()

#adpted form generate_training_data.py
if DO_FINAL_TEST:

  NUM_TARGET_VELOCITY_ITERS = 10
  
  #number of simulation steps per desired_v
  STEPS = 400
  
  NUM_DATAPTS = NUM_TARGET_VELOCITY_ITERS*STEPS
  
  #current datapt index
  datapt = 0
  
  #batch the real-time graph updates for better simulation preformance
  GRAPH_BATCH_SIZE = 100

  # Add the human-readable parts of the plot
  plt.figure(figsize=(7, 5))
  plt.title("PID Controller Output Over Time")
  plt.xlabel("Steps")
  plt.ion()

  #initialise the lines and then update the legend
  plt.plot([0], [0], color='black' , linewidth=1, label='Target Velocity (m/s)'             , dashes=[4,4], gapcolor="red")
  plt.plot([0], [0], color='green' , linewidth=2, label='PID Velocity Over Time (m/s)'      , dashes=[4,4], gapcolor="red")
  plt.plot([0], [0], color='yellow', linewidth=2, label='PID Acceleration Over Time (m/s^2)', dashes=[4,4], gapcolor="red")

  plt.plot([0], [0], color='black' , linewidth=1, label='Target Velocity (m/s)'               , dashes=[4,4], gapcolor="blue")
  plt.plot([0], [0], color='green' , linewidth=2, label='Model Velocity Over Time (m/s)'      , dashes=[4,4], gapcolor="blue")
  plt.plot([0], [0], color='yellow', linewidth=2, label='Model Acceleration Over Time (m/s^2)', dashes=[4,4], gapcolor="blue")
  
  plt.legend()
  plt.show()
  
  pid_car   = make_car(desired_v=20.0, dt=0.1)
  model_car = make_car(desired_v=20.0, dt=0.1)

  recorded_target             = np.zeros(NUM_DATAPTS, dtype = np.float32)
  pid_recorded_velocity       = np.zeros(NUM_DATAPTS, dtype = np.float32)
  pid_recorded_acceleration   = np.zeros(NUM_DATAPTS, dtype = np.float32)
  model_recorded_velocity     = np.zeros(NUM_DATAPTS, dtype = np.float32)
  model_recorded_acceleration = np.zeros(NUM_DATAPTS, dtype = np.float32)

  x_axis = np.linspace(1, NUM_DATAPTS, num=NUM_DATAPTS, dtype=int)

  for velocity in range(NUM_TARGET_VELOCITY_ITERS):
    pid_car  ["desired_v"] = np.random.uniform(2, 50)
    model_car["desired_v"] = pid_car["desired_v"]
    for step in range(STEPS):
      #code copieda form generate_training_data.py if you want proper explinatiopn that will be there.
      
      #run p controler to get error and theoreticaly desired acceleration
      (pid_acceleration, pid_error) = calculate_desired_acceleration(pid_car, K_P)
      pid_throttle_percentage = acceleration_to_throttle_percentage(pid_acceleration) #clip desired acceleration to be in [-1, 1]
      update(pid_car, pid_throttle_percentage) #update simulation

      #construct a tensor, plug it into the approxamator, convert output to float
      model_car_state = torch.tensor([[model_car["v"], model_car["desired_v"]]], dtype=torch.float32)
      with torch.inference_mode():
        model_acceleration = approx(model_car_state).item()

      #now busness as usual
      model_throttle_percentage = acceleration_to_throttle_percentage(model_acceleration)
      update(model_car, model_throttle_percentage) #update simulation

      recorded_target            [datapt] = pid_car["desired_v"]
      pid_recorded_velocity      [datapt] = pid_car["v"]
      pid_recorded_acceleration  [datapt] = pid_acceleration
      model_recorded_velocity    [datapt] = model_car["v"]
      model_recorded_acceleration[datapt] = model_acceleration
  
      #every GRAPH_BATCH_SIZE steps, graph: <velocty>, <error>, <target velocity>, <aceleration>
      if step % GRAPH_BATCH_SIZE == 0:
        plt.plot(x_axis[:datapt], recorded_target            [:datapt], color='black' , linewidth=1, label='Target Velocity (m/s)'               ,                              )
        plt.plot(x_axis[:datapt], pid_recorded_velocity      [:datapt], color='green' , linewidth=2, label='PID Velocity Over Time (m/s)'        , dashes=[4,4], gapcolor="red" )
        plt.plot(x_axis[:datapt], pid_recorded_acceleration  [:datapt], color='yellow', linewidth=2, label='PID Acceleration Over Time (m/s^2)'  , dashes=[4,4], gapcolor="red" )
        plt.plot(x_axis[:datapt], model_recorded_velocity    [:datapt], color='green' , linewidth=2, label='Model Velocity Over Time (m/s)'      , dashes=[4,4], gapcolor="blue")
        plt.plot(x_axis[:datapt], model_recorded_acceleration[:datapt], color='yellow', linewidth=2, label='Model Acceleration Over Time (m/s^2)', dashes=[4,4], gapcolor="blue")
        
        #pause graph until next update
        plt.pause(0.01)
  
      #move on to next datapt
      datapt += 1

input()