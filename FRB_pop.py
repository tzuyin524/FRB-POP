import sys
import numpy as np

# Returns the Gain in K/Jy for a position
# on the sky theta from boresight for a 
# radio telescope of diameter D (meter) operating
# at a wavelength lamb (meter) with an efficiency eta
def Gain(lamb, D, eta, theta):
    k = 1.38e-23 # boltszmann constant in Jouls/K
    FWHM = 1.22 * lamb/D # Beam 
    sigma = FWHM/2.355 # 2*np.sqrt(2*ln2)sigma = FWHM
    G0 = np.pi * D*D/4 * eta/2.0/k*1e-26
    #print("Gain is ",G0," K/Jy")
    value = G0 * np.exp(-1.0*theta*theta/2.0/sigma/sigma) # Beam ~ Gaussian
    return(value)

# radiometer equation to calculate SNR
def SNR(S,t,G,BW,Np,Trec,Tsky):
    return(S*G*np.sqrt(BW*t*Np)/(Trec+Tsky))
    
def main():
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("Error: Please provide an integer argument")
        print("Usage: python script.py <integer>")
        sys.exit(1)
    
    try:
        # Convert the argument to an integer
        number = int(sys.argv[1])
        
        # Do something with the number
        print(f"You entered: {number}")
        print(f"Double that is: {number * 2}")
        print(f"Squared is: {number ** 2}")
        R0_rad = np.pi/180 # radian
        lamb = 0.20 # meter
        D = 64.0 # meter
        S = 1 # Jy
        t = 0.001 # ms
        Np = 2
        Trec = 25 # K
        Tsky = 3
        BW = 340e6 # Hz

        for i in range(number):
            x = np.random.random() # <0-1)
            radius = R0_rad * np.sqrt(x) # uniform distribution over a circle
            #print("i = ",i, " radius = ",radius)
            G = Gain(lamb,D,0.6,radius)
            #print("The Gain was ",G," K/Jy")
            snr=SNR(S,t,G,BW,Np,Trec,Tsky)
            if (snr>1):
                print("snr is ",snr)
                print(x)
                # print("The Gain was ",G," K/Jy")

    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid integer")
        sys.exit(1)

if __name__ == "__main__":
    main()

