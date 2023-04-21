from periphery import PWM
import time

# Open PWM pin
pwm = PWM(1, 0)

pwm.frequency = 50
pwm.duty_cycle = 0.05

pwm.enable()

# Define pulse widths for left, middle, and right positions
left_pulse = 0.03
middle_pulse = 0.07
right_pulse = 0.11

try:
    while True:
        # Move to left position
        print('going left')
        pwm.duty_cycle = left_pulse
        time.sleep(1)

        # Move to right position
        print('going right')
        pwm.duty_cycle = right_pulse
        time.sleep(1)
except KeyboardInterrupt:
    pass

pwm.close()