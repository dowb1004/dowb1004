N = int(input())
numbers = input().split()
new_numbers =[]
for n in numbers:
    new_numbers.append(int(n))
v = int(input())


cnt = 0
for n in new_numbers:
    if n == v:
       cnt += 1 
print(cnt)