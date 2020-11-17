from smbus2 import SMBus
from mlx90614 import MLX90614
import webbrowser


# Do other stuff here...

# You can now close the image by doing

passedTemp = False
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)

while (not passedTemp):
    webbrowser.open(r"/home/pi/Desktop/covid19-screener-4317-g1/temp1.png")
    ambientTemp1 =  sensor.get_amb_temp()
    ambientTemp2 =  sensor.get_amb_temp()
    ambientTemp3 =  sensor.get_amb_temp()
    avg = (ambientTemp1 + ambientTemp2 + ambientTemp3 ) / 3
    print (avg)
    avgF = (avg * 1.8000) + 32
    print (sensor.get_obj_temp())
    bus.close()

    if (avgF > 99.0):
        #Display the image
        webbrowser.open(r"/home/pi/Desktop/covid19-screener-4317-g1/fail.png")
        break

    elif (avgF < 94.0):
         #Display the image
        webbrowser.open(r"/home/pi/Desktop/covid19-screener-4317-g1/temp2.png")
        image.show()
    else:
        #pass
        webbrowser.open(r"/home/pi/Desktop/covid19-screener-4317-g1/success.png")
        break
