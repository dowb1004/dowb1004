class Solution(object):
    def distanceBetweenBusStops(self, distance, start, destination):
        """
        :type distance: List[int]
        :type start: int
        :type destination: int
        :rtype: int
        """
        answer = 0
        forward = 0
        backward = 0
        
        if start < destination:
            for i in range(start, destination):
                forward += distance[i]
        else:
            for i in range(destination, start):
                forward += distance[i]

        backward = sum(distance) - forward
               
        if forward < backward:
            answer = forward
        else:
            answer = backward

        return answer
        