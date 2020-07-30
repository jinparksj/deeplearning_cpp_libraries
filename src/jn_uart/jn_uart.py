import serial
import time

sp = serial.Serial(port="dev/ttyTHS1", baudrate=9600, timeout=2)
time.sleep(1)
sp.flushOutput()
sp.flushInput()

try:
    if sp.open:
        while True:
            sp.write("UART test/".encode('utf-8'))
            print("sent")
            time.sleep(5)

except KeyboardInterrupt:
    print("Exit")

finally:
    sp.close()
    pass


