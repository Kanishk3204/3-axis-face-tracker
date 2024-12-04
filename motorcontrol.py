import RPi.GPIO as GPIO
import time

# Setup
GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
GPIO.setwarnings(False)

# Motor 1 (Horizontal)
IN1 = 17
IN2 = 27

# Motor 2 (Vertical)
IN3 = 22
IN4 = 23

# Motor 3 (Tilt)
IN5 = 5
IN6 = 6

# GPIO pins list
motor_pins = [IN1, IN2, IN3, IN4, IN5, IN6]

# Set all pins as output
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Function to run motor in forward direction
def motor_forward(in1, in2, duration):
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

# Function to run motor in reverse direction
def motor_reverse(in1, in2, duration):
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

# Test sequence
try:
    print("Testing Motor 1 (Horizontal)")
    motor_forward(IN1, IN2, 2)  # Run motor forward for 2 seconds
    motor_reverse(IN1, IN2, 2)  # Run motor backward for 2 seconds
    
    print("Testing Motor 2 (Vertical)")
    motor_forward(IN3, IN4, 2)  # Run motor forward for 2 seconds
    motor_reverse(IN3, IN4, 2)  # Run motor backward for 2 seconds
    
    print("Testing Motor 3 (Tilt)")
    motor_forward(IN5, IN6, 2)  # Run motor forward for 2 seconds
    motor_reverse(IN5, IN6, 2)  # Run motor backward for 2 seconds

finally:
    print("Cleaning up GPIO")
    GPIO.cleanup()  # Reset GPIO settings
