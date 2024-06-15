from djitellopy import Tello
import cv2
from ultralytics import YOLO
import torch
import time

#========================#
# Import the YOLO model  #
#========================#

model = YOLO('models/yolov8n.pt')
model_path = "models/R153tello_c_FIK.pt"  # Put the correct path to the downloaded model
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

#============================#
# Initialize the Tello drone #
#============================#

tello = Tello()
tello.connect()
battery_level = tello.get_battery()
tello.streamon()
frame_read = tello.get_frame_read()

#===================================#
# Coordinates of boxes to display   #
#===================================#

image_width = 960
image_height = 720
margin = 50
box_width = (image_width - 2 * margin) // 3

left_box = (margin, image_height // 6, image_width // 2, image_height - (image_height // 6))
right_box = (image_width // 2, image_height // 6, margin + 3 * box_width, image_height - (image_height // 6))

center_x_img = image_width // 2
center_y_img = image_height // 2

#===============#
# PI Parameters #
#===============#

Kp_X = 0.1
Ki_X = 0.0
Kp_Y = 0.2
Ki_Y = 0.0

Kp_XX = 0.09
Ki_XX = 0.00

#=========================#
# Initialize the errors   #
#=========================#

integral_X = 0
error_X = 0
previous_error_X = 0

integral_Y = 0
error_Y = 0
previous_error_Y = 0

integral_XX = 0
error_XX = 0
previous_error_XX = 0

integral_X_D = 0
error_X_D = 0
previous_error_X_D = 0
integral_Y_D = 0
error_Y_D = 0
previous_error_Y_D = 0

integral_X_G = 0
error_X_G = 0
previous_error_X_G = 0
integral_Y_G = 0
error_Y_G = 0
previous_error_Y_G = 0

#======================#
# Safety distance      #
#======================#
rotate_counter = 0
distance_security = 5
target_detected = False
tello.takeoff()
tello.move_down(30)
max_box_x_center = 0
max_box_y_center = 0

while True:
    img = frame_read.frame  # Get the video frames
    img = cv2.cvtColor(frame_read.frame, cv2.COLOR_RGB2BGR)
    new_frame_time = time.time()
    results = model(img)  # Return YOLOV8 results to results
    for result in results:  # For each detected object, get the bounding box (BBox) information
        boxes = result.boxes  # Get the coordinates of the BBox (bounding box)
    max_box_size = 0
    max_box_detection = None
    # Browse the BBox to find the largest in the frame

    for box in boxes:
        x = box.xyxy
        x = x.numpy()
        x = x.flatten()
        box_size = (x[2] - x[0]) * (x[3] - x[1])  # Calculate the area of each BBox in the frame

        if box_size > max_box_size:
            x1, y1, x2, y2 = x[:4]
            max_box_size = box_size
            max_box_detection = x
            max_box_height = max_box_detection[3] - max_box_detection[1]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Draw BBox for all detected objects in the frame
            proba = box.conf

        if (box.cls) == 39:  # Check if YOLOv8 detected a bottle class object in the frame, then load YOLOv5
            xx1, yy1, xx2, yy2 = x[:4]  # Check if YOLOv5 detected a bounding box which is our target
            target_detected = True
            box_target = x
            box_size = box_target[3] - box_target[1]
            cv2.putText(img, f"target detected", (int(xx1), int(yy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (0, 255, 0), 2)
        else:
            target_detected = False

    message = ""  # Empty message by default
    text_color = (255, 0, 0)  # Text color
    cv2.rectangle(img, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 255, 0), 2)  # Draw the left reference box
    cv2.rectangle(img, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 255, 0), 2)  # Draw the right reference box

    if max_box_detection is not None:  # Ensure there are detected objects

        left_box_size = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])  # Get the area of the left box
        right_box_size = (right_box[2] - right_box[0]) * (right_box[3] - right_box[1])  # Get the area of the right box

        tot_fit_left = max_box_size / left_box_size  # Calculate the total fit between the largest BBox in the frame and the left box
        tot_fit_right = max_box_size / right_box_size  # Calculate the total fit between the largest BBox in the frame and the right box

        left_box_x_center = (left_box[0] + left_box[2]) / 2  # Get the X coordinate of the left box center
        left_box_y_center = (left_box[1] + left_box[3]) / 2  # Get the Y coordinate of the left box center

        right_box_x_center = (right_box[0] + right_box[2]) / 2  # Get the X coordinate of the right box center
        right_box_y_center = (right_box[1] + right_box[3]) / 2  # Get the Y coordinate of the right box center

        max_box_x_center = (max_box_detection[0] + max_box_detection[2]) / 2  # Get the X coordinate of the largest BBox center in the frame
        max_box_y_center = (max_box_detection[1] + max_box_detection[3]) / 2  # Get the Y coordinate of the largest BBox center in the frame

        cv2.circle(img, (int(max_box_x_center), int(max_box_y_center)), 1, (0, 0, 255), 10)  # Draw a blue dot at the center of the largest BBox in the frame
        cv2.circle(img, (int(left_box_x_center), int(left_box_y_center)), 1, (255, 0, 0), 10)  # Draw a red dot at the center of the left box
        cv2.circle(img, (int(right_box_x_center), int(right_box_y_center)), 1, (255, 0, 0), 10)  # Draw a red dot at the center of the right box

        # Distance between the center of the largest BBox and the center of the left box
        dist_X_G = left_box_x_center - max_box_x_center
        dist_Y_G = left_box_y_center - max_box_y_center

        # Distance between the center of the largest BBox and the center of the right box
        dist_X_D = -(right_box_x_center - max_box_x_center)
        dist_Y_D = -(right_box_y_center - max_box_y_center)

        # Calculate the errors
        error_X_G = dist_X_G - distance_security
        error_Y_G = dist_Y_G - distance_security
        error_X_D = dist_X_D - distance_security
        error_Y_D = dist_Y_D - distance_security

    if target_detected:  # In case the target is detected in the frame

        #########################################################
        # Calculate the center of the BBox surrounding our target #
        #########################################################

        target_x_center = (box_target[0] + box_target[2]) / 2
        target_y_center = (box_target[1] + box_target[3]) / 2
        cv2.circle(img, (int(target_x_center), int(target_y_center)), 1, (0, 0, 255), 10)  # Draw a circle at the center of the BBox

        ############################################################################
        # Calculate the distance between the center of the frame and the center of the BBox #
        ############################################################################

        error_X = -(center_x_img - target_x_center)
        error_Y = (center_y_img - target_y_center)

        ########################################################
        # Estimate the distance between the target and the drone #
        ########################################################

        object_height_px = box_size  # Height in pixels of the BBox surrounding our target
        real_object_height_cm = 20  # Real height of the target in centimeters
        image_height_px = 720  # Image height in pixels (from preprocessing)
        focal_length_mm = 12.16  # Focal length of the camera in millimeters
        distance_mm = (real_object_height_cm * focal_length_mm * image_height_px) / (object_height_px * 10)

        ##########################################################################
        # Calculate the command to move the drone towards the target #
        ##########################################################################

        error_XX = distance_mm - 5

        integral_XX = integral_XX + error_XX
        uXX = Kp_XX * error_XX + Ki_XX * integral_XX
        previous_error_XX = error_XX

        ###########################################################################
        # Calculate the command to center the drone on the target along the x-axis  #
        ###########################################################################

        integral_X = integral_X + error_X
        uX = Kp_X * error_X + Ki_X * integral_X
        previous_error_X = error_X

        ###########################################################################
        # Calculate the command to center the drone on the target along the y-axis  #
        ###########################################################################

        integral_Y = integral_Y + error_Y
        uY = Kp_Y * error_Y + Ki_Y * integral_Y
        previous_error_Y = error_Y

        cv2.line(img, (int(center_x_img), int(center_y_img)), (int(center_x_img), int(center_y_img)), (255, 0, 0), 2)  # Draw a line between the center of the BBox and the center of the frame

        if isinstance(uX, torch.Tensor):
            uX = uX.item()  # Convert the Tensor to a scalar value
        if isinstance(uY, torch.Tensor):
            uY = uY.item()  # Convert the Tensor to a scalar value
        if isinstance(uXX, torch.Tensor):
            uXX = uXX.item()  # Convert the Tensor to a scalar value
        cv2.putText(img, f"Ux {uX}", (int(box_target[2]), int(box_target[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(img, f"Uy {uY}", (int(box_target[2]), int(box_target[3] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(img, f"Uxx {uXX}", (int(box_target[2]), int(box_target[3] - 50)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        if (max_box_size < left_box_size) and (max_box_size < right_box_size):  # If no obstacle, apply the command to move towards the target
            tello.send_rc_control(round(0.5 * uX), round(0.5 * uXX), round(0.5 * uY), round(0.5 * uX))
        else:  # Avoid the obstacle in front of the drone while trying to center towards the target to keep it in view
            if left_box[0] < max_box_x_center < left_box[2] and left_box[1] < max_box_y_center < left_box[3]:
                if (max_box_size > left_box_size) and (tot_fit_left >= 0.9) and proba > 0.8:
                    cv2.line(img, (int(left_box_x_center), int(left_box_y_center)), (int(max_box_x_center), int(max_box_y_center)), (255, 0, 0), 2)

                    integral_X_D = integral_X_D + error_X_D
                    uX_D = Kp_X * error_X_D + Ki_X * integral_X_D
                    previous_error_X_D = error_X_D

                    integral_Y_D = integral_Y_D + error_Y_D
                    uY_D = Kp_Y * error_Y_D + Ki_Y * integral_Y_D
                    previous_error_Y_D = error_Y_D

                    if isinstance(uX_D, torch.Tensor):
                        uX_D = uX_D.item()  # Convert the Tensor to a scalar value
                    if isinstance(uY_D, torch.Tensor):
                        uY_D = uY_D.item()  # Convert the Tensor to a scalar value

                    cv2.putText(img, f"Move rightttt {abs(uX_D)}", (int(max_box_detection[0]), int(max_box_detection[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                    tello.send_rc_control(round(-0.5 * uX_D), 0, round(-0.5 * uY_D), round(uX * -0.5))
            elif right_box[0] < max_box_x_center < right_box[2] and right_box[1] < max_box_y_center < right_box[3]:
                if (max_box_size > right_box_size) and (tot_fit_right >= 0.9):
                    cv2.line(img, (int(right_box_x_center), int(right_box_y_center)), (int(max_box_x_center), int(max_box_y_center)), (255, 0, 0), 2)

                    integral_X_G = integral_X_G + error_X_G
                    uX_G = Kp_X * error_X_G + Ki_X * integral_X_G
                    previous_error_X_G = error_X_G

                    integral_Y_G = integral_Y_G + error_Y_G
                    uY_G = Kp_Y * error_Y_G + Ki_Y * integral_Y_G
                    previous_error_Y_G = error_Y_G

                    if isinstance(uX_G, torch.Tensor):
                        uX_G = uX_G.item()  # Convert the Tensor to a scalar value
                    if isinstance(uY_G, torch.Tensor):
                        uY_G = uY_G.item()  # Convert the Tensor to a scalar value

                    cv2.putText(img, f"Move leftttt {abs(uX_G)}", (int(max_box_detection[0]), int(max_box_detection[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                    tello.send_rc_control(round(0.5 * uX_G), 0, round(0.5 * uY_G), round(uX * 0.5))
    else:  # If there is an obstacle and the target is absent, move along the x and y axes to avoid the obstacle
        if left_box[0] < max_box_x_center < left_box[2] and left_box[1] < max_box_y_center < left_box[3]:  # Check if the obstacle is on the left
            if (max_box_size > left_box_size) and (tot_fit_left >= 0.8):  # Check if the largest detected BBox is larger than the left reference box
                cv2.line(img, (int(left_box_x_center), int(left_box_y_center)), (int(max_box_x_center), int(max_box_y_center)), (255, 0, 0), 2)  # Draw a line between the center of the largest BBox and the center of the left reference box

                # Calculate the command to avoid the obstacle along the x and y axes

                integral_X_D = integral_X_D + error_X_D
                uX_D = Kp_X * error_X_D + Ki_X * integral_X_D
                previous_error_X_D = error_X_D

                integral_Y_D = integral_Y_D + error_Y_D
                uY_D = Kp_Y * error_Y_D + Ki_Y * integral_Y_D
                previous_error_Y_D = error_Y_D

                if isinstance(uX_D, torch.Tensor):
                    uX_D = uX_D.item()  # Convert the Tensor to a scalar value
                if isinstance(uY_D, torch.Tensor):
                    uY_D = uY_D.item()  # Convert the Tensor to a scalar value

                cv2.putText(img, f"Move right {abs(uX_D)}", (int(max_box_detection[0]), int(max_box_detection[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                tello.send_rc_control(round(-0.5 * uX_D), 0, round(-0.5 * uY_D), 0)
        elif right_box[0] < max_box_x_center < right_box[2] and right_box[1] < max_box_y_center < right_box[3]:
            if (max_box_size > right_box_size) and (tot_fit_right >= 0.8):
                cv2.line(img, (int(right_box_x_center), int(right_box_y_center)), (int(max_box_x_center), int(max_box_y_center)), (255, 0, 0), 2)

                integral_X_G = integral_X_G + error_X_G
                uX_G = Kp_X * error_X_G + Ki_X * integral_X_G
                previous_error_X_G = error_X_G

                integral_Y_G = integral_Y_G + error_Y_G
                uY_G = Kp_Y * error_Y_G + Ki_Y * integral_Y_G
                previous_error_Y_G = error_Y_G

                if isinstance(uX_G, torch.Tensor):
                    uX_G = uX_G.item()  # Convert the Tensor to a scalar value
                if isinstance(uY_G, torch.Tensor):
                    uY_G = uY_G.item()  # Convert the Tensor to a scalar value

                cv2.putText(img, f"Move left {abs(uX_G)}", (int(max_box_detection[0]), int(max_box_detection[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                tello.send_rc_control(round(0.5 * uX_G), 0, round(0.5 * uY_G), 0)
        else:
            tello.send_rc_control(0, 0, 0, 0)

    cv2.putText(img, f"Battery Level: {battery_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Inference", img)

    key = cv2.waitKey(1) & 0xff
    if key == 27:  # ESC
        tello.land()
        break

cv2.destroyAllWindows()
