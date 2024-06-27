import sys

input = lambda: sys.stdin.readline().rstrip()
print = lambda x: sys.stdout.write(str(x) + '\n')

N, M = map(int, input().split())  # map(.... 2개의 int 변환된 변수)
# print(N, M)
# number -> name
# name -> number
number_to_name = {}
name_to_number = {}
for i in range(N):
    name = input()
    # print(i, name)
    number = str(i + 1)
    # 굳이 M을 받을 때 변환할 필요 없이 여기서
    # number를 넣어버리자
    number_to_name[number] = name
    name_to_number[name] = number
# print(number_to_name) # 키 : 숫자, 값 : 문자열
# print(name_to_number) # 키 : 문자열, 값 : 숫자
for _ in range(M):
    new_input = input()
    # print(new_input)
    answer = number_to_name.get(new_input,
                       name_to_number.get(new_input))
    print(answer)