import scipy.stats 
import matplotlib.pyplot as plt
#------------------------------ Question 1 -------------------------#
# a
def statistics(li):
    mean = sum(li)/(l:=len(li)) # mean is sum/number of terms
    sd_sum = 0
    for i in li: # using a for loop to get all (values-mean)^2
        sd_sum+= (i-mean)**2
    var = sd_sum/l
    sd = var**(1/2)
    return mean, sd, l

# b 
def thresh(li, t):
    mean, sd, l = statistics(li)
    z = (t - mean)/sd # formula for z test 
    p_value = scipy.stats.norm.sf(abs(z)) # finding the p value/ likihood of 
    return z, p_value

def hypo(li, t):
    z, p_val = thresh(li, t)
    if z<0:
        return False
    return True

# c
class OOP:
    def __init__(self, file):
        self.values = [] 
        with open(file) as f:
            lines = [line.strip() for line in f]
            self.labels = (lines[0].split(','))[1:]
            for i in range(1, len(lines)):
                vals = lines[i].split(',')[1:]
                x=[float(i) for i in vals] 
                self.values.append(tuple(x))  
            
    def hypo(self, li, t):
        z, p_val = thresh(li, t)
        if z<0:
            return 0, p_val
        return 1, p_val
    
    def organize(self, t):
        results = []
        for i in self.values: 
            i = list(i)
            res, p = self.hypo(i, t)
            results.append((res, p))
        return results

#
def see_plot(li1, li2):
    plt.scatter(li1, li2)
    plt.xlabel("Gene Expression One")
    plt.ylabel("Gene Expression Two")
    plt.title("Scatterplot of Two Genes")
    plt.show()

    

    

data = [5.99342831, 4.7234714 , 6.29537708, 8.04605971, 4.53169325, 4.53172609, 8.15842563, 6.53486946, 4.06105123, 6.08512009]

print("#1a: ", statistics(data))
# z_score: negative means that the mean is above the threshold/fails, p value: Null hypo: our threshold equals the population mean, alt hyp: does not equal
print("#1b: ", thresh(data, 3)) # a = 0.05 if p>a then we fail to reject null hypo, else p<a we reject the null hypo
model = OOP("./technical_data/1_c_d.csv")
see_plot(model.values[0], model.values[1])