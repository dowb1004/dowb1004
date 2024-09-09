class Solution(object):
    def freqAlphabets(self, s):
        """
        :type s: str
        :rtype: str
        """
        d = {}
        for i in range(26):
            alphabet = chr(i + 97)
            num = i + 1
            if num < 10:
                d[str(num)] = alphabet
            else:
                d[str(num) + "#"] = alphabet
        sorted_keys = sorted(d.keys(), reverse=True, key=lambda x: (len(x), x))

        answer = ""
        while s:
            for key in sorted_keys:
                if s.startswith(key):                    
                    answer += d[key]
                    s = s[len(key):]
                    break 
       
        return answer
        