import serial
import time

sp = serial.Serial(port="/dev/ttyTHS1", baudrate=57600)
time.sleep(1)
sp.flushOutput()
sp.flushInput()

try:
    if sp.open:
        while True:
            sp.write("UART Test/".encode('utf-8'))
            print("sent")
            time.sleep(1)

except KeyboardInterrupt:
    print("Exit")

finally:
    sp.close()
    pass


