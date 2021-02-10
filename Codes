#1.1 Solve using scipy (command) optimization in python.
#importing necessary libraries
import numpy as np
from scipy.optimize import minimize
#defining objective
def objective(x):
x1=x[0]
x2=x[1]
x3=x[2]
return -3*x1-2*x2-4*x3
# initial guesses
x0=[0,0,0]
# show initial objective
print('Initial Objective: ' + str(objective(x0)))
#optimize

def cons1(x):
return -x[0]-x[1]-x[2]*2+4
def cons2(x):
return -2*x[0]-3*x[2]+5
def cons3(x):
return -2*x[0]-x[1]-3*x[2]+7
b = (0,10**5)
bnds = (b, b, b)
con1 = {'type': 'ineq','fun':cons1}
con2 = {'type': 'ineq','fun':cons2}
con3 = {'type': 'ineq','fun':cons3}
cons = [con1,con2,con3]
solution = minimize(objective,x0,method='SLSQP',\
bounds=bnds,constraints=cons)

x = solution.x
# show final objective
print('Final Objective: ' + str(objective(x)))
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))
print('x3 = ' + str(x[2]))

#1.2 Solve using GEKKO Library in python.
from gekko import GEKKO
import pandas as pd
import matplotlib.pyplot as plt
# Initialize Model
m = GEKKO(remote=True)
#help(m)
#define parameter
eq = m.Param(value=40)
#initialize variables
x1,x2,x3 = [m.Var() for i in range(3)]
#initial values
x1.value = 0
x2.value = 0
x3.value = 0
#lower bounds
x1.lower = 0
x2.lower = 0
x3.lower = 0
#Equations
m.Equation(x1 + x2+ 2*x3 <= 4)
m.Equation(2*x1 + 3*x3 <= 5)
m.Equation(2*x1 + x2 + 3*x3 <= 7)
#Objective

m.Obj(-3*x1 - 2*x2 - 4*x3)
#Set global options
m.options.IMODE = 3 #steady state optimization
#Solve simulation
m.solve() # solve on public server
#Results
print('')
print('Results')
print('x1: ' + str(x1.value))
print('x2: ' + str(x2.value))
print('x3: ' + str(x3.value))

#1.3: Develop a program to solve using simplex method.
import numpy as np
#generating matrix for all variables and equations
def gen_matrix(var,cons):
tab = np.zeros((cons+1, var+cons+2))
return tab
#finding the pivot column
def next_round_r(table):
m = min(table[:-1,-1])
if m>= 0:
return False
else:
return True
#checking if more pivots are requires except the bottom one
def next_round(table):
lr = len(table[:,0])
m = min(table[lr-1,:-1])
if m>=0:
return False
else:
return True
#finding the location of above listed elements
def find_neg_r(table):

lc = len(table[0,:])
m = min(table[:-1,lc-1])
if m<=0:
n = np.where(table[:-1,lc-1] == m)[0][0]
else:
n = None
return n
def find_neg(table):
lr = len(table[:,0])
m = min(table[lr-1,:-1])
if m<=0:
n = np.where(table[lr-1,:-1] == m)[0][0]
else:
n = None
return n
#finding pivot element corresponding to these values
def loc_piv_r(table):
total = []
r = find_neg_r(table)
row = table[r,:-1]
m = min(row)
c = np.where(row == m)[0][0]
col = table[:-1,c]
for i, b in zip(col,table[:-1,-1]):
if i**2>0 and b/i>0:
total.append(b/i)
else:
total.append(10000)
index = total.index(min(total))
return [index,c]
def loc_piv(table):
if next_round(table):
total = []
n = find_neg(table)
for i,b in zip(table[:-1,n],table[:-1,-1]):
if b/i >0 and i**2>0:
total.append(b/i)
else:
total.append(10000)
index = total.index(min(total))
return [index,n]

#finding pivot element through checking all the elements of the table and return a new table
def pivot(row,col,table):
lr = len(table[:,0])
lc = len(table[0,:])
t = np.zeros((lr,lc))
pr = table[row,:]
if table[row,col]**2>0:
e = 1/table[row,col]
r = pr*e
for i in range(len(table[:,col])):
k = table[i,:]
c = table[i,col]
if list(k) == list(pr):
continue
else:
t[i,:] = list(k-r*c)
t[row,:] = list(r)
return t
else:
print('Cannot pivot on this element.')
def convert(eq):
eq = eq.split(',')
if 'G' in eq:
g = eq.index('G')
del eq[g]
eq = [float(i)*-1 for i in eq]
return eq
if 'L' in eq:
l = eq.index('L')
del eq[l]
eq = [float(i) for i in eq]
return eq
#if we need to solve the minimization problem instead of maximization
def convert_min(table):
table[-1,:-2] = [-1*i for i in table[-1,:-2]]
table[-1,-1] = -1*table[-1,-1]
return table
#generation no. of variables
def gen_var(table):
lc = len(table[0,:])

lr = len(table[:,0])
var = lc - lr -1
v = []
for i in range(var):
v.append('x'+str(i+1))
return v
def add_cons(table):
lr = len(table[:,0])
empty = []
for i in range(lr):
total = 0
for j in table[i,:]:
total += j**2
if total == 0:
empty.append(total)
if len(empty)>1:
return True
else:
return False
#adding all the constraints
def constrain(table,eq):
if add_cons(table) == True:
lc = len(table[0,:])
lr = len(table[:,0])
var = lc - lr -1
j = 0
while j < lr:
row_check = table[j,:]
total = 0
for i in row_check:
total += float(i**2)
if total == 0:
row = row_check
break
j +=1
eq = convert(eq)
i = 0
while i<len(eq)-1:
row[i] = eq[i]
i +=1
row[-1] = eq[-1]
row[var+j] = 1

else:
print('Cannot add another constraint.')
#adding objective function
def add_obj(table):
lr = len(table[:,0])
empty = []
for i in range(lr):
total = 0
for j in table[i,:]:
total += j**2
if total == 0:
empty.append(total)
if len(empty)==1:
return True
else:
return False
#adding the objective function
def obj(table,eq):
if add_obj(table)==True:
eq = [float(i) for i in eq.split(',')]
lr = len(table[:,0])
row = table[lr-1,:]
i = 0
while i<len(eq)-1:
row[i] = eq[i]*-1
i +=1
row[-2] = 1
row[-1] = eq[-1]
else:
print('You must finish adding constraints before the objective function can be added.')
#the function is ready and will return the max variable in dic form
def maxz(table):
while next_round_r(table)==True:
table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)
while next_round(table)==True:
table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)
lc = len(table[0,:])
lr = len(table[:,0])
var = lc - lr -1
i = 0
val = {}

for i in range(var):
col = table[:,i]
s = sum(col)
m = max(col)
if float(s) == float(m):
loc = np.where(col == m)[0][0]
val[gen_var(table)[i]] = table[loc,-1]
else:
val[gen_var(table)[i]] = 0
val['max'] = table[-1,-1]
return val
#minima function
def minz(table):
table = convert_min(table)
while next_round_r(table)==True:
table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)
while next_round(table)==True:
table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)
lc = len(table[0,:])
lr = len(table[:,0])
var = lc - lr -1
i = 0
val = {}
for i in range(var):
col = table[:,i]
s = sum(col)
m = max(col)
if float(s) == float(m):
loc = np.where(col == m)[0][0]
val[gen_var(table)[i]] = table[loc,-1]
else:
val[gen_var(table)[i]] = 0
val['min'] = table[-1,-1]*-1
return val

#good to go and taking all the inputs
if __name__ == "__main__":
m = gen_matrix(3,3)
constrain(m,'1,1,2,L,4')
constrain(m,'2,0,3,L,5')
constrain(m,'2,1,3,L,7')
obj(m,'3,2,4,0')

print(maxz(m))
