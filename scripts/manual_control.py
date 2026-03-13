import airsim
import keyboard
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("--- Drone Manual Control Ready ---")
print("Controls:")
print("  T: Takeoff | L: Land | Esc: Emergency Land & Exit")
print("  W/S: Forward/Backward | A/D: Left/Right (Strafe)")
print("  Up/Down Arrows: Rise/Sink")
print("  Left/Right Arrows: Yaw (Rotate)")
print("----------------------------------")

# Configuration
speed = 5          # m/s
yaw_speed = 30     # deg/s (increased for better feel)
duration = 0.1     # Command duration

try:
    while True:
        # Emergency Quit
        if keyboard.is_pressed('esc'):
            print("Emergency stop triggered. Landing...")
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            break

        # Takeoff / Landing
        if keyboard.is_pressed('t'):
            print("Taking off...")
            client.takeoffAsync()
        elif keyboard.is_pressed('l'):
            print("Landing...")
            client.landAsync()

        # Initialize movement variables for this frame
        vx, vy, vz = 0, 0, 0
        yaw_rate = 0
        moving = False

        # Forward / Backward (X-axis in Body Frame)
        if keyboard.is_pressed('w'):
            vx = speed
            moving = True
        elif keyboard.is_pressed('s'):
            vx = -speed
            moving = True

        # Left / Right (Y-axis in Body Frame)
        if keyboard.is_pressed('a'):
            vy = -speed
            moving = True
        elif keyboard.is_pressed('d'):
            vy = speed
            moving = True

        # Up / Down (Z-axis in Body Frame: Negative is UP in NED)
        if keyboard.is_pressed('up'):
            vz = -speed
            moving = True
        elif keyboard.is_pressed('down'):
            vz = speed
            moving = True

        # Yaw (Rotation)
        if keyboard.is_pressed('left'):
            yaw_rate = -yaw_speed
            moving = True
        elif keyboard.is_pressed('right'):
            yaw_rate = yaw_speed
            moving = True

        # Execution
        if moving:
            # moveByVelocityBodyFrameAsync moves relative to where the drone is pointing
            client.moveByVelocityBodyFrameAsync(
                vx, vy, vz, 
                duration,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            )
        else:
            # If no keys are pressed, hover in place
            client.hoverAsync()

        # Small sleep to prevent high CPU usage
        time.sleep(0.02)

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Cleanup connection
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Program ended.")