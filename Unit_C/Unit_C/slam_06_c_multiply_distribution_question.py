# Multiply a distribution by another distribution.
# 06_c_multiply_distribution
# Claus Brenner, 26 NOV 2012
from pylab import plot, show
from distribution import *
import numpy as np

def multiply(a, b):
    """Multiply two distributions and return the resulting distribution."""

    # --->>> Put your code here.
    h_w_A = int(len(a.values)/2)
    h_w_B = int(len(b.values)/2)

    print(a.offset+h_w_A)
    print(b.offset+h_w_B)
    
    overlap_Start = max(a.offset, b.offset)
    overlap_Stop = min(a.offset+len(a.values), b.offset+len(b.values))

    print(overlap_Start)
    print(overlap_Stop)

    #A = a.values[np.argmax(a.values)-abs(a.offset-overlap_Start+h_w_A):np.argmax(a.values)+abs(a.offset-overlap_Stop+h_w_A)]
    #B = b.values[np.argmax(b.values)-abs(b.offset-overlap_Start+h_w_B):np.argmax(b.values)+abs(b.offset-overlap_Stop+h_w_B)]

    A = a.values[np.argmax(a.values)+(overlap_Start-a.offset-h_w_A):np.argmax(a.values)+(overlap_Stop-a.offset-h_w_A)]
    B = b.values[np.argmax(b.values)+(overlap_Start-b.offset-h_w_B):np.argmax(b.values)+(overlap_Stop-b.offset-h_w_B)]

    print(len(A))
    print(len(B))

    out = np.multiply(A,B)

    s = float(sum(out))
    if s != 0.0:
        out_val = [i / s for i in out]

    offset = overlap_Start
    d = Distribution(offset,out_val)
    return d  # Modify this to return your result.


if __name__ == '__main__':
    arena = (0,1000)

    # Here is our assumed position. Plotted in blue.
    position_value = 450
    position_error = 100
    position = Distribution.triangle(position_value, position_error)
    plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
         color='b', linestyle='--', drawstyle='steps')

    # Here is our measurement. Plotted in green.
    # That is what we read from the instrument.
    measured_value = 270
    measurement_error = 200
    measurement = Distribution.triangle(measured_value, measurement_error)

    plot(measurement.plotlists(*arena)[0], measurement.plotlists(*arena)[1],
         color='g', linestyle='--', drawstyle='steps')

    # Now, we integrate our sensor measurement. Result is plotted in red.
    position_after_measurement = multiply(position, measurement)
    plot(position_after_measurement.plotlists(*arena)[0],
         position_after_measurement.plotlists(*arena)[1],
         color='r', linestyle='--', drawstyle='steps')

    show()
