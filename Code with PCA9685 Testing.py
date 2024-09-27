import cv2
import mediapipe as mp
import math
import numpy as np
import adafruit_pca9685
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from board import SCL, SDA
import busio
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
norm = np.linalg.norm

#-----------------------------------------------------------------------------#
# Grid dimensions input
grid_dimensions_input = list(map(int,input("\nEnter dimensions (w x h --> e.g 8 x 6 = 8 6) : ").strip().split()))
grid_array = np.array(grid_dimensions_input)
grid_dimensions = grid_array - 1

#-----------------------------------------------------------------------------#
#Set up for arduino and servos
#connect Python to Arduino to PCA9685--> through serial port and I2C pin
i2c = busio.I2C(SCL, SDA)

board1 = adafruit_pca9685.PCA9685(i2c, address=0x40)
board2 = adafruit_pca9685.PCA9685(i2c, address=0x41)
board3 = adafruit_pca9685.PCA9685(i2c, address=0x42)

board1.set_pwm_freq(50)
board2.set_pwm_freq(50)
board3.set_pwm_freq(50)

#-----------------------------------------------------------------------------#
#Define rotation functions for each servo
def rotateservo(board,pin,angle):
        # Set the PWM value for the specified angle
        servo.Servo(board.channels[pin], min_pulse=500, max_pulse=2400)
        duty_cycle = int(4096 * ((angle * 11) + 500) / 20000)
        board.set_pwm(pin, 0, duty_cycle)

#-----------------------------------------------------------------------------#
#Functions
def distance_point_to_line(p_x, p_y, A, B, C):
    # Calculate the perpendicular distance from a point (p_x, p_y) to a line defined by Ax + By + C = 0.
    denominator = math.sqrt(A**2 + B**2)
    if denominator == 0 or math.isnan(denominator):
        return None
    else:
        dist = abs(A*p_x + B * p_y + C) / (denominator)
    return dist

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)
#-----------------------------------------------------------------------------#
# For webcam input:
cap = cv2.VideoCapture(0)

#-----------------------------------------------------------------------------#
#Change resolution to 1080p HD
make_1080p()

#Recode time between frames
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0
#-----------------------------------------------------------------------------#
#Conditions for hand tracking
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    success, image3 = cap.read()
    
#-----------------------------------------------------------------------------#
#If camera doesn't open --> safety
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
  
#-----------------------------------------------------------------------------#
#Reduce FPS
    new_fps = 60
    delay = 1/new_fps
    
    # time when we finish processing for this frame
    new_frame_time = time.time()
  
    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = int(fps)
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    fps = f"FPS : {fps}"
    
    # putting the FPS count on the window
    cv2.putText(image, fps, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 227, 252), 1, cv2.LINE_AA)
#-----------------------------------------------------------------------------#
#FIRST WINDOW (image)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lmList=[]
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          
     #Get data coordinates from hand landmarks
     #for loop gather data from landmarks --> need count and value from iterable
        for id, lm in enumerate(hand_landmarks.landmark):
             h, w, c = image.shape                                             #h = height to image, w = width of image
             cx = int(lm.x*w)                                                  #Coordinate x of landmark
             cy = int(lm.y*h)                                                  #Coordinate y of landmark 
             lmList.append([id,cx,cy])
             lm_text = f"{id, cx, cy}"
             lm_text_width, lm_text_height = cv2.getTextSize(lm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
             cv2.putText(image, lm_text, (cx - lm_text_width// 2, cy  + lm_text_height * 2 ),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
             # print(id, cx, cy)                                                 #id = which landmark it is 
                                                                               #lm = landmark coordinates
     #Draw hand marks and hand connections           
      mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      
#SECOND WINDOW (image3)
      cv2.rectangle(image3,(0,0),(1920,1080),(0,0,0),-1)
      mp_drawing.draw_landmarks(
           image3,
           hand_landmarks,
           mp_hands.HAND_CONNECTIONS,
           mp_drawing_styles.get_default_hand_landmarks_style(),
           mp_drawing_styles.get_default_hand_connections_style())
      cv2.putText(image3, fps, (30,3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 227, 252), 1, cv2.LINE_AA)
      
#-----------------------------------------------------------------------------#   
    # Draw lines for grid
      for x in range(1,grid_dimensions[0]+1):
            box_x_pos = int((x * w) / (grid_dimensions[0]+1))
            cv2.line(image, (box_x_pos, 0), (box_x_pos, h), (0, 0, 255), 1) 
            cv2.line(image3, (box_x_pos, 0), (box_x_pos, h), (0, 0, 255), 1)
                        
      for y in range(1,grid_dimensions[1]+1):
            box_y_pos = int((y * h) / (grid_dimensions[1]+1))
            cv2.line(image, (0, box_y_pos), (w, box_y_pos), (0, 0, 255), 1)
            cv2.line(image3, (0, box_y_pos), (w, box_y_pos), (0, 0, 255), 1)

#-----------------------------------------------------------------------------#
    # Box center coordinates 
      box_centers = []                                                         # Define list to hold box center points
    # Calculate box center points and draw centre point
      for y in range(grid_dimensions[1]+1):
          for x in range(grid_dimensions[0]+1):
              box_x_pos = int((w/(2 * grid_array[0])) * ((2 * x) + 1))
              box_y_pos = int((h/(2 * grid_array[1])) * ((2 * y) + 1))
              box_centers.append((box_x_pos, box_y_pos))
              box_centres_text = f"{box_x_pos , box_y_pos}"
              box_centres_text_width, box_centres_text_height = cv2.getTextSize(box_centres_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
              cv2.putText(image3, box_centres_text, (box_x_pos - box_centres_text_width// 2, box_y_pos  + box_centres_text_height * 2 ),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
              #Draw circle for centre point
              cv2.circle(image3,(box_x_pos, box_y_pos ), 3, (255,0,0), -1)    
              cv2.circle(image3,(box_x_pos, box_y_pos ), 60, (255,0,0), 2) 
              cv2.circle(image,(box_x_pos, box_y_pos ), 60, (255,0,0), 2) 
      box_centers_array = np.array(box_centers)                                # set up box centers as an array   

#-----------------------------------------------------------------------------#
#Distance between each hand landmark --> ie hand connection length
      #Calculate distance from each landmark to the next wrt the context of each connection in diagram 
      handconnectiondistance0to4 = []
      handconnectiondistance5to8 = []
      handconnectiondistance9to12 = []
      handconnectiondistance13to16 = []
      handconnectiondistance17to20 = []
       # Distances in fingers
      for o in range(0,4):
           distance = math.sqrt((lmList[o + 1][1] - lmList[o][1])**2 + (lmList[o + 1][2] - lmList[o][2])**2)
           handconnectiondistance0to4.append(distance)
      for o in range(5,8):
           distance = math.sqrt((lmList[o + 1][1] - lmList[o][1])**2 + (lmList[o + 1][2] - lmList[o][2])**2)
           handconnectiondistance5to8.append(distance) 
      for o in range(9,12):
           distance = math.sqrt((lmList[o + 1][1] - lmList[o][1])**2 + (lmList[o + 1][2] - lmList[o][2])**2)
           handconnectiondistance9to12.append(distance)
      for o in range(13,16):
           distance = math.sqrt((lmList[o + 1][1] - lmList[o][1])**2 + (lmList[o + 1][2] - lmList[o][2])**2)
           handconnectiondistance13to16.append(distance)
      for o in range(17,20):
           distance = math.sqrt((lmList[o + 1][1] - lmList[o][1])**2 + (lmList[o + 1][2] - lmList[o][2])**2)
           handconnectiondistance17to20.append(distance)
       #Distances from 0-5, 5-9, 9-13, 13-17, 0-17,     
      distance_0_5 = math.sqrt((lmList[5][1] - lmList[0][1])**2 + (lmList[5][2] - lmList[0][2])**2)
      distance_5_9 = math.sqrt((lmList[9][1] - lmList[5][1])**2 + (lmList[9][2] - lmList[5][2])**2)
      distance_9_13 = math.sqrt((lmList[13][1] - lmList[9][1])**2 + (lmList[13][2] - lmList[9][2])**2)
      distance_13_17 = math.sqrt((lmList[17][1] - lmList[13][1])**2 + (lmList[17][2] - lmList[13][2])**2)
      distance_0_17 = math.sqrt((lmList[17][1] - lmList[0][1])**2 + (lmList[17][2] - lmList[0][2])**2)
       
      #Reorganise the order of the HNAD CONNECTIONS into correct order
      hand_connection_list = list(mp_hands.HAND_CONNECTIONS)
      hand_connection_list = [hand_connection_list[10], hand_connection_list[12], hand_connection_list[18], hand_connection_list[0],
                              hand_connection_list[1], hand_connection_list[7], hand_connection_list[16], hand_connection_list[20],
                              hand_connection_list[8], hand_connection_list[11], hand_connection_list[14], hand_connection_list[19],
                              hand_connection_list[13], hand_connection_list[4], hand_connection_list[9], hand_connection_list[17],
                              hand_connection_list[5], hand_connection_list[3], hand_connection_list[2], hand_connection_list[6],
                              hand_connection_list[15]]
      
      handconnectiondistance = []
      for connection in hand_connection_list:
          x1, y1 = lmList[connection[0]][1], lmList[connection[0]][2]
          x2, y2 = lmList[connection[1]][1], lmList[connection[1]][2]
          distance1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
          distance1 = round(distance1, 2)
          handconnectiondistance.append(distance1)
          # Display distance next to hand connections
          text = f"{distance1}"
          text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
          cv2.putText(image3, text, ((x1 + x2) // 2 - text_width // 2, (y1 + y2) // 2 + text_height // 2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
          cv2.circle(image,(x2, y2), 5, (255,0,0), -1)
            
#-----------------------------------------------------------------------------#
    #Calculate distances between box centers and hand landmarks or hand connections
      distances_dictionary = {}             #Dictionary of distances and respective point distance is calculated from
      distances = []                        #Full list of distances every 0-20 --> lm1 to box 1, 20-40 --> perpendicular distance from connection1 to box 1
      connection_dist = []
      connection_dist_dictionary = {}
      landmark_dist = []
      hand_connection_line_eq = {}
      box_center_check = []
      
      triangle_sideA = []
      am1 = []
      az1 = []
      bz1 = []
      tri_angle = []
       # Define the radius within which to draw perpendicular lines
      threshold = 60
      radius = threshold
      
      for i, box_center in enumerate(box_centers_array):
              for j, lm in enumerate(lmList):
                    lm_dist = math.sqrt((box_center[0] - lm[1])**2 + (box_center[1] - lm[2])**2)
                    landmark_dist.append(lm_dist)
                    distances.append(lm_dist)
                    if lm_dist <= radius:
                        # Draw a line from the landmark point to the center point if within radius
                        cv2.line(image, (lm[1], lm[2]), (box_center[0], box_center[1]), (0, 255, 0), 1)
                    key = f"Box {i+1} to Landmark {j+1}"
                    distances_dictionary[key] = (lm_dist, lmList[connection[0]], lmList[connection[1]])
                   # distances --> gives a dictionary of the distances perpedenicularly from lm or connection
                   
#-----------------------------------------------------------------------------#      
      #Create matrix size of grid to for on and off servos
      #Switch numbers in grid array --> [2 1] into [1 2]
      matrix_grid = [grid_array[1], grid_array[0]]
      one_zero_servo = np.zeros(matrix_grid, dtype=float)
      min_dist_servo = np.zeros(matrix_grid, dtype=float)
      min_dist_list = []
      
      #New distance list --> every cell responds to each box_center and has its own list of 40 (1st 20 is the distance from lm, 2nd 20 is perpendicular distance)
      full_distances_list = [] #List of sublists containing all distances to each respective box_center
      for i in range(0 , len(distances), 21):
          full_distances_list.append(distances[i:i+21])
    
      #attempt 2 with finding minimum distance in sublist to see if smaller than threshold
      full_distance_matrix_sublist = np.zeros(matrix_grid, dtype=list)
      for i in range(matrix_grid[0]):
            for j in range(matrix_grid[1]):
                distances_sublist = full_distances_list[i * matrix_grid[1] + j]
                val = min(distances_sublist)
                r_up_val = round(val,3)
                min_dist_servo[i][j] = r_up_val
                min_dist_list.append(r_up_val)
                full_distance_matrix_sublist[i][j] = distances_sublist
                if val < threshold:
                    one_zero_servo[i][j] = 1
                    
                else:
                        one_zero_servo[i][j] = 0
      print(one_zero_servo)
      
      one_zero_list = []
      for row in one_zero_servo:
          for element in row:
                  one_zero_list.append(element)
      # print(min_dist_servo) 
#-----------------------------------------------------------------------------#
#Delay the video feed
      time.sleep(delay)  
#-----------------------------------------------------------------------------# 
      #Servo activation from matrix input    
      # Define the number of rows and columns in the matrix
      num_rows_1_0_servo = len(one_zero_servo)
      num_cols_1_0_servo = len(one_zero_servo[0])

      num_servos_per_row = grid_dimensions_input[0]
      num_servos_per_cols = grid_dimensions_input[1]

      Board1_ServoPins = []    #*
      Board2_ServoPins = []
      Board3_ServoPins = []
      count = 0
      for i in range(0,15):
          servo.Servo(board1.channels[i], min_pulse=500, max_pulse=2400)
          servo.Servo(board2.channels[i], min_pulse=500, max_pulse=2400)
          servo.Servo(board3.channels[i], min_pulse=500, max_pulse=2400)
          servo[i] = servo.Servo(board3.channels[i], min_pulse=500, max_pulse=2400)
          #Might be this ^^
          Board1_ServoPins.append(i)
          Board2_ServoPins.append(i)
          Board3_ServoPins.append(i)
         
    
    
      min_distance_list = []
      for row in min_dist_servo:
          for element in row:
                  min_distance_list.append(element)

      servos = [0 for _ in range(num_servos_per_row * num_servos_per_cols)]
      angle_matrix = np.zeros(matrix_grid, dtype=float)
      #Loop through the one_zero_servo matrix
      count = 0
                      
      for i in range(len(one_zero_servo)):
            for j in range(len(one_zero_servo[i])):
                count +=1
                print(count)
                print(one_zero_servo[i][j])
                print(i)
                print(j)
                if one_zero_servo[i][j] == 1:
                      # Turn on the corresponding LED
                      # servos[i * num_servos_per_cols + j][0]                  #ON
                      angle = 180 - ((180/threshold) * min_dist_servo[i][j])
                      angle_matrix[i][j] = angle
                      rotateservo(servo_pins_matrix[j][i],angle_matrix[i][j])
#-----------------------------------------------------------------------------#
    # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands 1', cv2.flip(image, 1))
      cv2.imshow('MediaPipe Hands 1', image)
      # cv2.imshow('Validation window', cv2.flip(image3, 1))
      cv2.imshow('Validation window Flipped', image3)
    
#-----------------------------------------------------------------------------#    
    # Set up "esc" key to end video capture and close windows
    if cv2.waitKey(5) & 0xFF == 27:
      break
board1.deinit()
board2.deinit()
board3.deinit()
cap.release()
cv2.destroyAllWindows() 
