# Histogram implementation of a bayes filter - combines
# convolution and multiplication of distributions, for the
# movement and measurement steps.
# 06_d_histogram_filter
# Claus Brenner, 28 NOV 2012
from pylab import plot, show, ylim
from distribution import *
import numpy as np

def move(distribution, delta):
    """Returns a Distribution that has been moved (x-axis) by the amount of
       delta."""
    return Distribution(distribution.offset + delta, distribution.values)



# --->>> Copy your convolve(a, b) and multiply(a, b) functions here.
def multiply(a, b):
    """Multiply two distributions and return the resulting distribution."""

    # --->>> Put your code here.
    h_w_A = int(len(a.values)/2)
    h_w_B = int(len(b.values)/2)

    #print(a.offset+h_w_A)
    #print(b.offset+h_w_B)
    
    overlap_Start = max(a.offset, b.offset)
    overlap_Stop = min(a.offset+len(a.values), b.offset+len(b.values))

    #print(overlap_Start)
    #print(overlap_Stop)

    A = a.values[np.argmax(a.values)+(overlap_Start-a.offset-h_w_A):np.argmax(a.values)+(overlap_Stop-a.offset-h_w_A)]
    B = b.values[np.argmax(b.values)+(overlap_Start-b.offset-h_w_B):np.argmax(b.values)+(overlap_Stop-b.offset-h_w_B)]

    #print(len(A))
    #print(len(B))

    out = np.multiply(A,B)

    s = float(sum(out))
    if s != 0.0:
        out_val = [i / s for i in out]

    offset = overlap_Start
    d = Distribution(offset,out_val)
    return d  # Modify this to return your result.

def convolve(a, b):
    """Convolve distribution a and b and return the resulting new distribution."""

    # --->>> Put your code here.
    
    size = len(a.values) + len(b.values) - 1
    val = [0]*(size)
    offset = a.offset + b.offset

    for i in range(len(a.values)):
        for j in range(len(b.values)):
            val[i+j] = val[i+j] + a.values[i]*b.values[j]

    return Distribution(offset, val)


if __name__ == '__main__':
    arena = (0,220)

    # Start position. Exactly known - a unit pulse.
    start_position = 10
    position = Distribution.unit_pulse(start_position)
    plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
         linestyle='--', drawstyle='steps')

    # Movement data.
    controls  =    [ 20 ] * 10

    # Measurement data. Assume (for now) that the measurement data
    # is correct. - This code just builds a cumulative list of the controls,
    # plus the start position.
    p = start_position
    measurements = []
    for c in controls:
        p += c
        measurements.append(p)

    # This is the filter loop.
    for i in range(len(controls)):
        # Move, by convolution. Also termed "prediction".
        control = Distribution.triangle(controls[i], 10)
        position = convolve(position, control)
        plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
             color='b', linestyle='--', drawstyle='steps')

        # Measure, by multiplication. Also termed "correction".
        measurement = Distribution.triangle(measurements[i], 10)
        position = multiply(position, measurement)
        plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
             color='r', linestyle='--', drawstyle='steps')

    show()
