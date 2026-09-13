import torch 

def rowswap(matrix, source_row, target_row):
    
    #will swap two rows in a matrix
    
    #make copy of the original matrix 
    result = matrix.clone()
    
    #save the source row in temporary variable 
    temp = result[source_row].clone()
    
    #swap the rows 
    result[source_row] = result[target_row]
    result[target_row] = temp

    return result


def rowscale(matrix, source_row, scaling_factor):
    
    #will scale a row in a matrix by a scaling factor 
    
    #make copy of the original matrix
    result = matrix.clone()
    
    #multiply the row by the scaling factor
    result[source_row] = result[source_row]*scaling_factor
    
    return result

def rowreplacement(matrix, first_row, second_row, j, k):
    
    #will replace a row in a matrix with the sum of itself and a scaled version of another row
    #jR_i + kR_j
    
    #make copy of the original matrix 
    result = matrix.clone()
    
    #replace the target row with the sum of itself and the scaled source row
    result[second_row] = j*result[first_row] + k*result[second_row]
    
    return result


def rref (matrix):
    
    #will return the reduced row echelon form of a matrix 
    #Row Echelon Form where each Pivot element is 1
    
    #make copy of the original matrix but in float format
    result = matrix.clone().to(torch.float64)
    
    #number of rows and columns in the matrix
    num_rows, num_cols = result.shape
    
    #row where the next pivot will be placed
    pivot_row = 0
    
    #iterate through each column of the matrix
    for col in range(num_cols):
        
        #will stop when every row has received a pivot 
        
        if pivot_row >= num_rows:
            break
        
        #first we will search for a nonzero pivot in current column 
        pivot_candidate = None
        
        for row in range(pivot_row, num_rows):
            
            if not torch.isclose(result[row, col], torch.tensor(0.0, dtype=torch.float64)):
                pivot_candidate = row
                break
            
        
        #if whole column is zero below pivot row we move to the next column
        if pivot_candidate is None:
            continue
        
        #swap the pivot candidate in correct row 
        
        if pivot_candidate != pivot_row:
            result = rowswap(result, pivot_candidate, pivot_row)
            
        
        #get current pivot value
        pivot_value = result[pivot_row, col]
        
        #scale the pivot row to make pivot value 1 
        result = rowscale(result, pivot_row, 1.0/pivot_value)
        
        #make every value below pivot zero 
        for row in range(pivot_row + 1, num_rows):
            
            value_below_pivot = result[row, col]
            
            if not torch.isclose(value_below_pivot, torch.tensor(0.0, dtype=result.dtype)):
                
                #make value value_below_pivot zero
                result = rowreplacement(result, pivot_row, row, -value_below_pivot, 1.0)
                
        
        #increment pivot row for next iteration
        pivot_row += 1

    # Remove very small floating-point values
    result[torch.abs(result) < 1e-10] = 0.0
       
    return result


#test cases
if __name__ == "__main__":
    
    print("Test Row Swap:")
    
    A = torch.tensor([
        [1.0, 3.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 1.0, -4.0]
    ])
    
    print("Original Matrix:")
    print(A)
    
    print("After Swapping Row 1 and Row 2:")
    A1 = rowswap(A, 0, 1)
    print(A1)
    
    #test row scaling
    print("\nTest Row Scaling:")
    
    
    print("After Scaling (1/3)Row1 :")
    
    A2 = rowscale(A1, 0, 1.0/3.0)
    print(A2)
    
    #test row replacement
    print("\nTest Row Replacement:")
    
    print("After R3 = -3R1 + R3: ")
    
    A3 = rowreplacement(A2, 0, 2, -3.0, 1.0)
    print(A3)
    
    
    #test rref
    B = torch.tensor([
        [0.0, 3.0, -6.0, 6.0, 4.0],
        [3.0, -7.0, 8.0, -5.0, 8.0],
        [3.0, -9.0, 12.0, -9.0, 6.0]
    ])    
    print("\nTest RREF:")
    
    print("Original Matrix B:")
    print(B)
    
    print("Reduced Row Echelon Form:")
    B1 = rref(B)
    print(B1)