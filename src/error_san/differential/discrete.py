

def diff(t, x):
    v = []
    
    #check if the lengths of both the arrays match or not
    if len(x) != len(t):
        print("ERROR: Length mismatch")
        return None
    #else we move forward and do the calculation
    else: 
        for i in range(1, len(x)):
            #here we compute the discrete derivative between two consecutive samples and store it in v
            v.append((x[i] - x[i-1])/(t[i] - t[i-1]))
        return v