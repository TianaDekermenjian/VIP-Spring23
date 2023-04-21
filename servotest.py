from periphery import PWM

# Open PWM chip 0, channel 10
pwm = PWM(2, 0)

# Set frequency to 50 Hz
pwm.frequency = 50
# Set duty cycle to 5%
pwm.duty_cycle = 0.05

pwm.enable()

pwm.close()