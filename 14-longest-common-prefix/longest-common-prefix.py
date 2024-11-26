class Solution(object):
    def longestCommonPrefix(self, strs):

        answer = ""
        pre_string = strs[0]
        flag = True
        if len(strs) == 1:
            return strs[0]

        for i in range(1, len(strs)):
            tmp = ""
            string = strs[i]
            flag = True
            if len(pre_string) <= len(string):
                string_len = len(pre_string)
            else:
                string_len = len(string)

            for j in range(string_len):
                pre_char = pre_string[j]                                                
                char = string[j]
                if pre_char == char:
                    tmp += pre_char
                else:                           
                    break
            pre_string = tmp
            answer = tmp                    
                        
        return answer
        