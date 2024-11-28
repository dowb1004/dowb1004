class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if s.count("(") != s.count(")"):
            return False
        if s.count("[") != s.count("]"):
            return False
        if s.count("{") != s.count("}"):
            return False
 
        answer = False        
        tmp = []
        for i, c in enumerate(s):
            if c == "(" or c == "{" or c == "[":
                 tmp.append(c)
            else:
                if tmp:
                    if c == ")" and tmp[-1] == "(":
                        tmp.pop()
                    elif c == "}" and tmp[-1] == "{":
                        tmp.pop()
                    elif c == "]" and tmp[-1] == "[":
                        tmp.pop()

        if tmp:
            answer = False
        else:
            answer = True

        return answer
        