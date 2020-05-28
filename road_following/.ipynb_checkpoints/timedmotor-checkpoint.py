

from jetbot import Robot
import time


robot = Robot()


speed = 0.40#ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, description='speed gain')
total_time = 0

for count in range(100):

    start_time = time.time()
        # angle_last = angle    
    robot.left_motor.value = speed
    robot.right_motor.value = speed

    timeOneIteration = time.time() - start_time 
    total_time += timeOneIteration
    print(count)


    time.sleep(0.01)

    

print("Average Time: ",total_time / 100)
robot.stop()