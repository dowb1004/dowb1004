class Solution(object):
    def checkString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        answer = True
        appear_b = False
        for c in s:
            if c == "a" and appear_b == True:
                # a가 등장했는데 appear_b가 True라면 정답은 False!
                answer = False
            elif c == "b":
                appear_b = True
        print("appear_b", appear_b)
        return answer
        