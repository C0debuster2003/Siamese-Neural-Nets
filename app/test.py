count = 0
n = int(input())
my_list = []
x = 2
while(count!=n):
    fact_count = 0
    for i in range(1,x+1):
        if(x%i==0):
            fact_count+=1
    if(fact_count == 2):
        count+=1
        my_list.append(x)
    if(count == n):
        break
    x+=1
print(my_list)