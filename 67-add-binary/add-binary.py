class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        int_a = int(a)
        int_b = int(b)

        int_sum = int_a + int_b
        str_sum = list(str(int_sum))

        c = ""
        pre = 0
        for i in range(len(str_sum)-1, -1, -1):
            if pre == 0 and str_sum[i] == "2":
                pre = 1
                str_sum[i] = "0"
            elif pre == 1 and str_sum[i] == "2":
                pre = 1
                str_sum[i] = "1"
            elif pre == 1 and str_sum[i] == "1":
                pre = 1
                str_sum[i] = "0"
            elif pre == 1 and str_sum[i] == "0":
                pre = 0
                str_sum[i] = "1"
            

        if pre == 1:
            c = "1" + "".join(str_sum)
        else:
            c = "".join(str_sum)

        return c


        